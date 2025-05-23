import copy
import os
from collections import OrderedDict

import arg_parser
# import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
import transform_dataset

# import pruner
from trainer import validate, IgnoreBaseline, RecalibrationBaseline


def main(args=None, wandb_run=None, model_trained=None):
    if args is None:
        args = arg_parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("Method: " + args.unlearn + ", Device: " + str(device))

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.to(device)

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        print(len(forget_dataset))
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        print(len(retain_dataset))
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader, full=train_loader_full
    )

    trained_model = None

    if args.task == "replace" and args.replacement_type == "plausibly_incorrect_label":
        checkpoint = torch.load("./trained_models/0cifar100_resnet18_model_SA_best.pth.tar", map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        trained_model = copy.deepcopy(model).to(device)

        trained_model.load_state_dict(checkpoint, strict=False)

    normalize = None

    if args.task == "replace":
        unlearn_data_loaders = utils.replace_data(unlearn_data_loaders, args, model=trained_model)
        if args.replacement_type == "domain_adaptation":
            unlearn_data_loaders, normalize = unlearn_data_loaders
        retain_dataset = unlearn_data_loaders["retain"].dataset
        forget_dataset = unlearn_data_loaders["forget"].dataset

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")

    del trained_model

    """
    print('val dataset:')
    for i, (image, target) in enumerate(val_loader):
        print(target)
    
    print('test dataset:')   
    for i, (image, target) in enumerate(test_loader):
        print(target)
    """

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain" and args.unlearn != "retrain_until":
            print("Loading trained model")
            model.load_state_dict(checkpoint, strict=False)
        else:
            print("Retraining model")

        evals = None

        # Start a timer
        if device != torch.device("cpu"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        if args.unlearn != "Renorm" and args.unlearn != "Platt":
            unlearn_method = unlearn.get_unlearn_method(args.unlearn)

            if normalize is not None:
                print("Replacing model.normalize for domain adaptation task")
                model.normalize = normalize.to(device)

            if model_trained is None:
                output = unlearn_method(data_loaders=unlearn_data_loaders, model=model, criterion=criterion, args=args, optimizer=None, epoch=0, mask=None,
                                   wandb_run=wandb_run)
                
                if args.unlearn == "fisher" or args.unlearn == "wfisher" or args.unlearn == "FF":
                    model = copy.deepcopy(evals).to(device)

                    evals = None
                else:
                    if args.unlearn in ["retrain", "SSD", "SCRUB", "EUk", "CUk"]:
                        new_model, evals = output
                        model = new_model
                    else:
                        evals = output
            else:
                checkpoint = model_trained
                if "state_dict" in checkpoint.keys():
                    checkpoint = checkpoint["state_dict"]

                print("Loading trained model")
                model.load_state_dict(checkpoint, strict=False)
                evals = None
                new_model = model

        epoch_count = evals["epoch_count"] if (evals is not None and "epoch_count" in evals) else 0

        # End the timer
        if device != torch.device("cpu"):
            end.record()
        # unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    accuracy = {}
    log_model = None

    if "new_accuracy" not in evaluation_result:
        for name, loader in unlearn_data_loaders.items():
            print(name)
            utils.dataset_convert_to_test(loader.dataset, args)
            if args.unlearn == "Renorm":
                val_acc, loss = IgnoreBaseline(loader, model, criterion, args)
            elif args.unlearn == "Platt":
                val_acc, loss, log_model = RecalibrationBaseline(loader, unlearn_data_loaders["test"], model, criterion, args, "platt", log_model=log_model)
            else:
                val_acc, loss = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")
            if args.task == "replace" and args.replacement_type == "poisoning" and name != "forget":
                poisoned_loader = transform_dataset.inject_backdoor_loader(loader, args)
                utils.dataset_convert_to_test(poisoned_loader.dataset, args)
                val_acc, loss = validate(loader, model, criterion, args)
                accuracy[f"{name}_poisoned"] = val_acc
                print(f"{name}_poisoned acc: {val_acc}")

        if torch.cuda.is_available():
            accuracy['time'] = start.elapsed_time(end) / 1000
        accuracy['epoch_count'] = epoch_count
        accuracy['log_model'] = log_model

        evaluation_result["accuracy"] = accuracy
        # unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    metrics = {
        "log": evals,
        "final": accuracy,
    }

    if args.unlearn in ["retrain", "SSD", "SCRUB", "EUk", "CUk"]:
        return metrics, new_model
    else:
        return metrics, model


if __name__ == "__main__":
    main()

    # python -u main_forget.py  --save_dir './_results/forget/class/seed3/FT' --mask './temp_results/cifar10/origin/0model_SA_best.pth.tar' --unlearn FT --class_to_replace 0 --seed 3 --dataset cifar10 --unlearn_epochs 1 --unlearn_lr 0.1
