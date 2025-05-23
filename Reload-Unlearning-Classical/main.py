import copy

import wandb

import arg_parser

import torch
import torch.nn as nn

import utils
import transform_dataset

import os
import pickle

import main_train
import main_forget, main_random
import random

from trainer import validate, IgnoreBaseline, RecalibrationBaseline
import knowledge_values as kv

def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    configs_dir = "./configs"

    os.makedirs(configs_dir, exist_ok=True)

    utils.setup_seed(args.seed)
    seed = args.seed

    args.backdoor_dir = f"./backdoor_logs"
    if args.task == "replace":
        args.backdoor_dir += f"/{args.backdoor_type if args.replacement_type == 'poisoning' else '_'}_{args.sweep_dataset}_{args.seed}"
        os.makedirs(args.backdoor_dir, exist_ok=True)

    args.device = device

    if args.sweep_dataset is not None:
        args.dataset = args.sweep_dataset

    (
        trained_model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    trained_model.to(device)

    dataset_size = len(train_loader_full.dataset)

    if args.num_indexes_to_replace is None:
        # Randomly select portion of the dataset to replace
        args.num_indexes_to_replace = int(args.prop_forget * dataset_size)

    if args.indexes_to_replace is None:
        # Randomly select the indexes to replace
        args.indexes_to_replace = torch.randperm(dataset_size)[: args.num_indexes_to_replace]

    if args.task == "replace" and args.replacement_type == "poisoning":
        class_lim = 10
        if args.dataset == "cifar100":
            class_lim = 100
        options = [i for i in range(0, class_lim)]
        # Shuffle clss
        clss = random.sample(options, 2)

        # Randomly select indexes from the same class
        indices = []
        targets = train_loader_full.dataset.targets if args.dataset != "svhn" else train_loader_full.dataset.labels
        for i in range(dataset_size):
            if targets[i] in clss:
                indices.append(i)

        args.indexes_to_replace = indices
        args.num_indexes_to_replace = len(indices)

    args.class_to_replace = None

    trained_model_dir = "./trained_models"
    log_dir = "./running_logs"

    args.save_dir = trained_model_dir

    os.makedirs(trained_model_dir, exist_ok=True)

    adder = args.num_indexes_to_replace if args.task != "replace" else 0
    trained_model_filename = f"{args.arch}_{seed}_{args.dataset}_{args.replacement_type}_{adder}"
    trained_model_path = trained_model_dir + f"/0{args.arch}_{seed}_{args.dataset}_{args.replacement_type}_{adder}"

    if args.task == "replace" and args.replacement_type in ['noisy', 'poisoning']:
        if args.replacement_type == 'noisy':
            trained_model_path += f"_{args.noise_type}_{args.noise_level}"
            trained_model_filename += f"_{args.noise_type}_{args.noise_level}"
        else:
            trained_model_path += f"_{args.backdoor_type}"
            trained_model_filename += f"_{args.backdoor_type}"

    trained_model_path += ".pth.tar"
    trained_model_filename += ".pth.tar"

    print(os.path.exists(trained_model_path))

    exp_name = "Sample Experiment Name"
    if args.replacement_type == "noisy" and args.task == "replace":
        args.noise_level = 0.4
        args.noise_type = "gaussian"
        exp_name = "Higher Noise 0.4"

    wandb_run = wandb.init(
        project="RELOAD-machine-unlearning",
        dir="./wandb",

        config={
            "unlearn": args.unlearn,
            "unlearn_lr": args.unlearn_lr,
            "arch": args.arch,
            "num_indexes_to_replace": args.num_indexes_to_replace,
            "seed": args.seed,
            "dataset": args.dataset,
            "Experiment Name": exp_name,
        },

        settings=wandb.Settings(_disable_stats=True)
    )

    # Check if the trained model exists already
    if os.path.exists(trained_model_path):
        print("RELEARNING")
        if torch.cuda.is_available():
            job_dir = os.environ.get("SLURM_JOB_ID")

        # Load the trained model
        args.mask = trained_model_path
        print(f"MASK: {args.mask}")

        if args.task == "replace" and args.replacement_type == "noisy":
            wandb_run.log({"noise type": args.noise_type,
                           "noise level": args.noise_level})

        run_id = wandb.run.id

        # Get the classes from which datapoints are being removed
        if (args.replacement_type in ["feature_removal", "feature_invariance", "domain_adaptation"]) and args.task == "replace":
            threshold_value = args.threshold
        else:
            threshold_value = max(min(args.threshold * args.num_indexes_to_replace / dataset_size, 1.0), 0.05)

        wandb.log({"threshold_value": threshold_value})

        args.threshold = threshold_value

        # Generate a mask for the indexes to replace
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

        # Generate a mask for the indexes to replace
        data_loaders, forget_gradients = kv.main(run_id, args)

        args.forget_gradients = forget_gradients

        # End the timer and log the time it took to generate the mask
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            mask_generation_time = start.elapsed_time(end) / 1000
            wandb.log({"kv calculation time": mask_generation_time})

        # Change the path to a run-specific directory
        args.path = os.path.join(args.save_dir, run_id + " with_{" + str(threshold_value) + "}.pt")

        unlearning_method_names = ["retrain", args.unlearn]

        unlearning_methods = [
            [unlearning_method_names[i], main_forget.main, None] if unlearning_method_names[i] != args.unlearn else
            [unlearning_method_names[i], main_random.main, None] for i in range(len(unlearning_method_names))]

        # Create a dict to store the results of each unlearning method
        metrics = {}
        logs = {}

        # Create a dict to store the models of each unlearning method
        models = {}

        for key in forget_gradients.keys():
            forget_gradients[key] = forget_gradients[key].to("cpu")

        process_iter = [(method, args, forget_gradients, wandb_run) for method in unlearning_methods]

        unlearn_to_epoch = {
            "GA": 20,
            "FT": 50,
            "wfisher": 192,
            "retrain": 182,
            "fisher": 192,
            "GKT": 92,
            "RELOAD": 35,
            "SSD": 0,
        }

        if args.task == "replace":
            if args.replacement_type in ["noisy", "plausibly_incorrect_label", "incorrect_label", "targeted_label_flip", "poisoning"]:
                # path = log_dir + f"/{args.num_indexes_to_replace}_{args.replacement_type}_{args.noise_type}_{args.noise_level}_{args.seed}_retrain.pt"
                path = f"{trained_model_dir}/0{args.dataset}_{args.arch}_model_SA_best.pth.tar"
            else:
                path = log_dir + f"/{args.num_indexes_to_replace}_{args.replacement_type}_{args.dataset}_{args.seed}_retrain.pt"
        else:
            path = log_dir + f"/{args.num_indexes_to_replace}_{args.dataset}_{args.seed}_retrain.pt"

        results = [None for _ in range(0, len(process_iter))]
        for i in range(0, len(process_iter)):
            if process_iter[i][0][0] == "retrain" and torch.cuda.is_available() and os.path.exists(path):
                with open(path, "rb") as f:
                    results[i] = torch.load(f)
            else:
                process_iter[i][1].unlearn_epochs = unlearn_to_epoch[process_iter[i][0][0]]
                results[i] = utils.multi_func(process_iter[i])
                if process_iter[i][0][0] == "retrain":
                    if not torch.cuda.is_available():
                        args.until_threshold = results[i][1][0]['final']['retain']
                    with open(path, "wb") as f:
                        torch.save(results[i], f)

        for i in range(0, len(results)):
            # This should be the same length as unlearning methods
            # method = unlearning_methods[i][0]
            try:
                method, outcome = results[i]
            except:
                # Then the retrained method is the trained model and doesnt have all the other expected values
                results[i] = utils.multi_func(process_iter[i], results[i])
                method, outcome = results[i]
            metric, model = outcome
            models[method] = model
            metrics[method] = metric["final"]
            logs[method] = metric["log"]

        # Define the different comparison metrics
        comparison_metrics = ['full', 'forget', 'retain', 'val', 'test']

        if args.task == "replace" and args.replacement_type == "poisoning":
            for metric in comparison_metrics:
                if metric != "forget":
                    poisoned_loader = transform_dataset.inject_backdoor_loader(data_loaders[metric], args)
                    data_loaders[f"{metric}_poisoned"] = poisoned_loader

            comparison_metrics.extend(["retain_poisoned", "val_poisoned", "test_poisoned"])


        # Log all of the results
        for method in metrics.keys():
            for metric in comparison_metrics:
                wandb.log({
                    f"{method}_{metric}_accuracy": metrics[method][metric],
                })

        # Log the time taken for each process
        for method in metrics.keys():
            if method != "original" and torch.cuda.is_available():
                wandb.log({
                    f"{method}_time": metrics[method]["time"],
                })

        # Send all models to the device
        for key in models.keys():
            models[key] = models[key].to(device)

        for mod in models.keys():
            # Plot the decision boundary if the dataset is synthetic
            # if args.dataset == "2d_synthetic":
            #     utils.plot_decision_boundary(models[mod], mod, data_loaders, "forget", device, args, True)
            test_set_copy = copy.deepcopy(data_loaders["test"])
            losses = utils.compute_losses(models[mod], data_loaders["forget"], device)
            scrub_forget_mia = utils.scrub_mia(models[mod], test_set_copy, data_loaders["forget"], args, False)
            wandb.log({f"{mod}_scrubbed_MIA_forget": scrub_forget_mia.mean(),
                       f"{mod}_forget_loss": losses.mean()})

        # Compare the Symmetric KL Divergences pairwise between the models
        kl_divergences = {}
        for metric in comparison_metrics:
            for mod1 in models.keys():
                for mod2 in models.keys():
                    if mod1 != mod2 and f"{mod1}_{mod2}_{metric}_kl_divergence" not in kl_divergences.keys() and f"{mod2}_{mod1}_{metric}_kl_divergence" not in kl_divergences.keys():
                        # Perform an average KL over the entire metric loader
                        forward_kl_running_sum = []
                        reverse_kl_running_sum = []
                        symmetric_kl_running_sum = []
                        for i, data in enumerate(data_loaders[metric]):
                            image, target = data
                            image = image.to(device)
                            target = target.to(device)

                            output1 = models[mod1](image)
                            output2 = models[mod2](image)

                            # check if output is a tuple
                            if isinstance(output1, tuple):
                                output1 = output1[0]
                            # check if output is a tuple
                            if isinstance(output2, tuple):
                                output2 = output2[0]

                            kl_divergence = torch.nn.functional.kl_div(
                                torch.nn.functional.log_softmax(output1, dim=1),
                                torch.nn.functional.softmax(output2, dim=1),
                                reduction="batchmean")
                            reverse_kl = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(output2, dim=1),
                                                                    torch.nn.functional.softmax(output1, dim=1),
                                                                    reduction="batchmean")
                            forward_kl_running_sum.append(kl_divergence.item())
                            reverse_kl_running_sum.append(reverse_kl.item())
                            symmetric_kl_running_sum.append((kl_divergence + reverse_kl).item())
                        kl_divergences[f"{mod1}_{mod2}_{metric}_skl_divergence"] = sum(
                            symmetric_kl_running_sum) / len(
                            symmetric_kl_running_sum)
                        kl_divergences[f"{mod1}_{mod2}_{metric}_forward_kl_divergence"] = sum(
                            forward_kl_running_sum) / len(
                            forward_kl_running_sum)
                        kl_divergences[f"{mod2}_{mod1}_{metric}_reverse_kl_divergence"] = sum(
                            reverse_kl_running_sum) / len(
                            reverse_kl_running_sum)

        # Log the KL Divergences
        for key, value in kl_divergences.items():
            wandb.log({key: value})

        # Cleanup created files
        os.remove(args.path)

        # Finish the wandb run
        wandb.finish()

        # Exit the program
        return
    else:
        print("TRAINING")
        if args.replacement_type == "plausibly_incorrect_label":
            checkpoint = torch.load("./trained_models/0cifar100_resnet18_model_SA_best.pth.tar", map_location=device)
            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]

            trained_model.load_state_dict(checkpoint, strict=False)

        if args.task == "replace":
            train_loader_full = utils.replace_data(train_loader_full, args, model=trained_model)

        # Run the training regime
        main_train.main(trained_model_filename, train_loader_full, args, wandb_run)

        # DUMP A FILE SAYING THIS IS COMPLETE
        os.makedirs("./logs", exist_ok=True)
        with open(f"./logs/0{trained_model_filename}_MARKER.pkl", "wb+") as f:
            pickle.dump(True, f)

        # Exit the program
        return


if __name__ == "__main__":
    main()
    # The first time this script is run - if the trained model file does not exist it will train it
    # Run the script again with the same parameters and it will unlearn the model