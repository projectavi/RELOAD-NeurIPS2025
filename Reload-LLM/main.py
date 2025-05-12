import torch
from transformers import GenerationConfig

from eco.model import HFModel
from cft import send_requests, parse_responses, load_cft_from_cache
import time
import argparse
import evaluate_module
from tqdm import tqdm

from eco.utils import seed_everything

import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Unlearn a prompt from a model")
    parser.add_argument("--model", type=str, default="phi-1_5", help="Path to the model")
    parser.add_argument("--prompt", type=str, default="Who is Darth Vader?", help="Prompt to unlearn")
    parser.add_argument("--retrain_dataset", type=str, default=None, help="Path to the dataset of retained data to retrain on")
    parser.add_argument("--alr", type=float, default=1e-2, help="Ascent Learning rate for unlearning")
    parser.add_argument("--threshold", type=float, default=0.01, help="Threshold for unlearning")
    parser.add_argument("--target_layer_key", type=str, default="self_attn.q_proj.weight", help="Target layers to unlearn")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--forget_split", type=str, default="forget01", help="Split to use for forgetting")
    parser.add_argument("--retain_split", type=str, default="retain99", help="Split to use for retaining")
    parser.add_argument("--holdout_split", type=str, default="real_authors", help="Split to use for testing")
    parser.add_argument("--model_config_dir", type=str, default="model_config", help="Directory to save the model")
    parser.add_argument("--replacement_strategy", type=str, default="mean", help="Replacement strategy for unlearning")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for finetuning")
    parser.add_argument("--ft_epochs", type=int, default=1, help="Number of epochs for finetuning")
    parser.add_argument("--ft_optim", type=str, default="SGD", help="Optimizer for finetuning")
    parser.add_argument("--ascent_optim", type=str, default="SGD", help="Optimizer for unlearning ascent")
    parser.add_argument("--retain_sample_size", type=int, default=100, help="Sample size for finetuning on the retain dataset")  
    parser.add_argument("--ft_all_params", type=str, default="false", help="Train all parameters")
    args = parser.parse_args()
    return args

def select_params(model, target_layer_key):
    selected_params = []
    for name, param in model.named_parameters():
        if target_layer_key in name:
            selected_params.append(name)
            param.requires_grad = True
        else:
            param.requires_grad = False
    return selected_params

def apply_chat_template(model, prompt):
    # prompt_with_chat_template = model.tokenizer.apply_chat_template(
    #     [{"role": "user", "content": prompt}],
    #     tokenize=False,
    #     add_generation_prompt=True,
    # )
    # return prompt_with_chat_template
    return prompt

def generate_model_output(model, prompt):
    return model.generate(
        **model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
                model.device
            ),
            generation_config=model.generation_config,
            eos_token_id=model.tokenizer.eos_token_id,
    )

def decode_model_output(model, output):
    return model.tokenizer.batch_decode(output, skip_special_tokens=False)[0]

def format_cft_prompt(prompt_with_cft_unformatted):
    prompt_with_cft = prompt_with_cft_unformatted
    for k in range(len(prompt_with_cft_unformatted)):
        if prompt_with_cft_unformatted[k] == ":":
            prompt_with_cft = prompt_with_cft_unformatted[k + 1:]
            break
    return prompt_with_cft

def print_model_params(model):
    for name, param in model.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

def evaluate_model_all(model, forget_dataset, retain_dataset, real_authors_dataset, world_facts_dataset, forget_perturbed_dataset, retain_perturbed_dataset, real_authors_perturbed_dataset, world_facts_perturbed_dataset, args, wandb_run=None, eval_handle=""):
    print(f"Evaluating {eval_handle} model")

    print(f"Forget Dataset Evaluations:")
    # forget_answer_prob = evaluate_module.answer_prob(model, forget_dataset)
    # print(f"Forget Dataset Answer Probability: {forget_answer_prob}")
    # forget_truth_ratio = evaluate_module.truth_ratio(model, forget_dataset, forget_perturbed_dataset)
    # print(f"Forget Dataset Truth Ratio: {forget_truth_ratio}")
    # forget_rougel = evaluate_module.rouge_l(model, forget_dataset, args)
    # print(f"Forget Dataset ROUGE-L: {forget_rougel}")
    forget_quality = evaluate_module.forget_quality(model, forget_dataset, forget_perturbed_dataset, args)
    print(f"Forget Quality: {forget_quality}")
    
    # Slice the retain dataset into smaller batches for evaluation as its too large to load into memory
    retain_dataset_sample = retain_dataset.shuffle(seed=args.seed)
    retain_dataset_sample = retain_dataset_sample.select(range(0, 100))

    model_utility, retain_eval, real_authors_eval, world_facts_eval = evaluate_module.model_utility(
        model, retain_dataset_sample, real_authors_dataset, world_facts_dataset, retain_perturbed_dataset, real_authors_perturbed_dataset, world_facts_perturbed_dataset, args
    )  

    print(f"Model Utility: {model_utility}")
    print(f"Retain Dataset Evaluations:")
    print(f"Retain Dataset Answer Probability: {retain_eval['probability']}")
    print(f"Retain Dataset Truth Ratio: {retain_eval['truth_ratio']}")
    print(f"Retain Dataset ROUGE-L: {retain_eval['rouge_l']}")

    print(f"Real Authors Dataset Evaluations:")
    print(f"Real Authors Dataset Answer Probability: {real_authors_eval['probability']}")
    print(f"Real Authors Dataset Truth Ratio: {real_authors_eval['truth_ratio']}")
    print(f"Real Authors Dataset ROUGE-L: {real_authors_eval['rouge_l']}")

    print(f"World Facts Dataset Evaluations:")
    print(f"World Facts Dataset Answer Probability: {world_facts_eval['probability']}")
    print(f"World Facts Dataset Truth Ratio: {world_facts_eval['truth_ratio']}")
    print(f"World Facts Dataset ROUGE-L: {world_facts_eval['rouge_l']}")

    combination = model_utility + forget_quality
    print(f"Objective Function: {combination}")

    if wandb_run is not None:
        wandb_run.log({
            # f"{eval_handle}_forget_answer_prob": forget_answer_prob,
            # f"{eval_handle}_forget_truth_ratio": forget_truth_ratio,
            # f"{eval_handle}_forget_rougel": forget_rougel,
            f"{eval_handle}_forget_quality": forget_quality,
            f"{eval_handle}_model_utility": model_utility,
            f"{eval_handle}_retain_answer_prob": retain_eval["probability"],
            f"{eval_handle}_retain_truth_ratio": retain_eval["truth_ratio"],
            f"{eval_handle}_retain_rougel": retain_eval["rouge_l"],
            f"{eval_handle}_real_authors_answer_prob": real_authors_eval["probability"],
            f"{eval_handle}_real_authors_truth_ratio": real_authors_eval["truth_ratio"],
            f"{eval_handle}_real_authors_rougel": real_authors_eval["rouge_l"],
            f"{eval_handle}_world_facts_answer_prob": world_facts_eval["probability"],
            f"{eval_handle}_world_facts_truth_ratio": world_facts_eval["truth_ratio"],
            f"{eval_handle}_world_facts_rougel": world_facts_eval["rouge_l"],
            f"{eval_handle}_objective": combination,
        })

    return combination

def main_full_batch(model, args, forget_dataset, retain_dataset, real_authors_dataset, world_facts_dataset, wandb_run=None):
    selected_params = select_params(model.model, args.target_layer_key)

    percent_params = len(selected_params) / len(list(model.model.named_parameters()))
    print(f"Percent of parameters selected for unlearning: {percent_params * 100:.2f}%")
    if wandb_run is not None:
        wandb_run.log({
            "percent_params": percent_params,
        })

    print(f"Selected {len(selected_params)} parameters for unlearning") 

    embedding_size = model.model.get_input_embeddings().weight.shape[0]
    if len(model.tokenizer) > embedding_size:
        model.model.resize_token_embeddings(len(model.tokenizer))

    compiled_knowledge_inputs = []
    prompts_with_chat_template = []
    prompts_with_chat_template_output_texts = []

    for prompt in forget_dataset['prompt']:
        prompt_with_chat_template = apply_chat_template(model, prompt)

        prompts_with_chat_template.append(prompt_with_chat_template)

    batch_ids, uncached_prompts = load_cft_from_cache(prompts_with_chat_template)
    BATCH_SIZE = len(prompts_with_chat_template)

    uncached_batch_ids = []

    if len(uncached_prompts) != 0:
        print("Not all prompts are CFT cached, generating them")
        for prompt_with_chat_template in uncached_prompts:
            prompt_with_chat_template_output = generate_model_output(model, prompt_with_chat_template)
            prompt_with_chat_template_output_text = decode_model_output(model, prompt_with_chat_template_output)[len(prompt_with_chat_template_output) :]

            prompts_with_chat_template_output_texts.append(prompt_with_chat_template_output_text)

        uncached_batch_ids = send_requests(uncached_prompts, prompts_with_chat_template_output_texts)

    batch_ids = batch_ids + uncached_batch_ids
        
    try:
        results = parse_responses(batch_ids)
    except Exception as e:
        print(f"Awaiting processing: {e}")
        time.sleep(300)
        results = parse_responses(batch_ids)

    start = time.time()

    wrapped_results = tqdm(range(len(results)), desc="Processing CFT", unit="batch")
    for i in wrapped_results:
        prompt_with_cft_unformatted = results[i]["response"]
        prompt_with_cft = format_cft_prompt(prompt_with_cft_unformatted)

        # print(f"Prompt with CFT: {prompt_with_cft}")

        prompt_with_cft_and_question = prompt_with_cft + prompt
        prompt_with_cft_and_question_and_chat_template = apply_chat_template(model, prompt_with_cft_and_question)

        extracted_knowledge = generate_model_output(model, prompt_with_cft_and_question_and_chat_template)
        extracted_knowledge_text = decode_model_output(model, extracted_knowledge)[len(prompt_with_cft_and_question_and_chat_template) :]

        # print(f"Extracted knowledge: {extracted_knowledge_text}")

        extracted_knowledge_inputs = model.tokenizer(extracted_knowledge_text, return_tensors="pt").to(args.device)
        extracted_knowledge_inputs["labels"] = extracted_knowledge_inputs["input_ids"]

        compiled_knowledge_inputs.append(extracted_knowledge_inputs)

    # Now we want to unlearn this knowledge

    print(f"Total parameters for optimization: {len([param for name, param in model.model.named_parameters() if name in selected_params])}")
    optim_fn = getattr(torch.optim, args.ascent_optim)
    optimizer = optim_fn([param for name, param in model.model.named_parameters() if name in selected_params], lr=args.alr)
    
    for name, param in model.model.named_parameters():
        if name in selected_params:
            param.requires_grad = True
        else:
            param.requires_grad = False

    retain_dataset_sample = retain_dataset.shuffle(seed=args.seed)
    retain_dataset_sample = retain_dataset_sample.select(range(0, args.retain_sample_size))

    retain_dataset_sample_prompts = retain_dataset_sample["prompt"]
    retain_dataset_sample_answers = retain_dataset_sample["answer"]

    retain_sample_inputs = [p + a for p, a in zip(retain_dataset_sample_prompts, retain_dataset_sample_answers)]

    model.model.zero_grad()

    wrapped_retain_inputs = tqdm(retain_sample_inputs, desc="Retain Set Computation", unit="batch")
    for input in wrapped_retain_inputs:
        encoding = model.tokenizer(input, padding="longest", return_tensors="pt").to(
            args.device
        )
        encoding["labels"] = encoding["input_ids"]
        output = model(**encoding)
        loss = output.loss
        loss.backward()
    forget_gradients = {}   
    for name, param in model.model.named_parameters():
        if name in selected_params:
            if param.grad is not None:
                forget_gradients[name] = param.grad.clone().to("cpu")

    model.model.zero_grad()
    wrapped_inputs = tqdm(compiled_knowledge_inputs, desc="Unlearning", unit="batch")
    for input in wrapped_inputs:
        output = model(**input)
        loss = output.loss
        loss.backward()

    for name, param in model.model.named_parameters():
        if name in selected_params:
            if param.grad is not None:
                forget_gradients[name] = (torch.abs(param.grad.clone().to("cpu")) + 1e-10) / (torch.abs(param.grad.clone().to("cpu") + forget_gradients[name]) + 1e-10)

                # Flip for ascent with optimizer
                param.grad = -param.grad / BATCH_SIZE

    print(f"Keys in forget gradients: {len(forget_gradients.keys())}")


    optimizer.step()
    optimizer.zero_grad()

    print(f"Ascent step completed")

    model.model.to("cpu")
    
    all_gradients = [tensor.flatten().to(args.device) for tensor in forget_gradients.values()]
    all_gradients = torch.cat(all_gradients).to("cpu")

    print(f"Total number of parameters in forget gradients: {all_gradients.numel()}")

    threshold = torch.max(torch.topk(all_gradients, int(args.threshold * len(all_gradients)), largest=False).values)

    if args.replacement_strategy == "mean":
        fn = torch.mean
    elif args.replacement_strategy == "zero":
        fn = torch.zeros_like
    elif args.replacement_strategy == "random":
        fn = torch.randn_like
    elif args.replacement_strategy == "uniform":
        fn = lambda x: torch.empty_like(x).uniform_(-1, 1)
    elif args.replacement_strategy == "normal":
        fn = lambda x: torch.empty_like(x).normal_()
    elif args.replacement_strategy == "kaiming_uniform":
        fn = torch.nn.init.kaiming_uniform_
        backup_fn = lambda x: torch.empty_like(x).uniform_(-1, 1)
    elif args.replacement_strategy == "kaiming_normal":
        fn = torch.nn.init.kaiming_normal_
        backup_fn = lambda x: torch.empty_like(x).normal_()
    elif args.replacement_strategy == "xavier_uniform":
        fn = torch.nn.init.xavier_uniform_
        backup_fn = lambda x: torch.empty_like(x).uniform_(-1, 1)
    elif args.replacement_strategy == "xavier_normal":
        fn = torch.nn.init.xavier_normal_
        backup_fn = lambda x: torch.empty_like(x).normal_()
    else:
        raise ValueError(f"Unknown replacement strategy: {args.replacement_strategy}")

    wrapped_named_params = tqdm(model.model.named_parameters(), desc="Resetting parameters", unit="layer")
    for name, param in wrapped_named_params:
        if name in forget_gradients and name in selected_params:
            # print(f"Resetting {forget_gradients[name][forget_gradients[name] < threshold].numel()} parameters in layer {name}")
            if args.replacement_strategy in ["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal"] and len(param.shape) < 2:
                param.data = torch.where(forget_gradients[name] < threshold,  backup_fn(param.data), param.data)
            else:
                param.data = torch.where(forget_gradients[name] < threshold,  fn(param.data), param.data)

    model.model.to(args.device)

    end = time.time()
    print(f"Unlearning completed in {end - start} seconds")

    if wandb_run is not None:
        wandb_run.log({
            "unlearning_time": end - start,
        })

    return model

def finetune_model(model, dataset, args, all_params=False, wandb_run=None):

    start = time.time()

    optim_fn = getattr(torch.optim, args.ft_optim)

    if all_params:
        model.model.train()
        optimizer = optim_fn(
            model.model.parameters(),
            lr=args.lr,
        )

    else:
        selected_params = select_params(model.model, args.target_layer_key)
        print(f"Selected {len(selected_params)} parameters for finetuning")
        optimizer = optim_fn(
            [param for name, param in model.model.named_parameters() if name in selected_params],
            lr=args.lr,
        )
 

    model.model.zero_grad()

    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.select(range(0, args.retain_sample_size))

    prompts = dataset["prompt"]
    answers = dataset["answer"]

    inputs = [p + a for p, a in zip(prompts, answers)]

    for epoch in range(args.ft_epochs):
        epoch_avg_loss = 0
        wrapped_inputs = tqdm(inputs, desc="Finetuning", unit="batch")
        for input in wrapped_inputs:
            encoding = model.tokenizer(input, padding="longest", return_tensors="pt").to(
               args.device
            )
            encoding["labels"] = encoding["input_ids"]
            output = model(**encoding)
            loss = output.loss
            epoch_avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        epoch_avg_loss /= len(encoding)
        print(f"Epoch {epoch + 1}/{args.ft_epochs}, Loss: {epoch_avg_loss}")
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "loss": epoch_avg_loss,
            })

    end = time.time()
    print(f"Finetuning completed in {end - start} seconds")
    if wandb_run is not None:
        wandb_run.log({
            "finetuning_time": end - start,
        })

        # wandb_run.log({
            # "total_reload_time": wandb_run.history["unlearning_time"][-1] + wandb_run.history["finetuning_time"][-1],
        # })

    return model

def setup_tofu(args):
    from eco.dataset import TOFU
    dataset = TOFU()
    forget_dataset = dataset.load_dataset_for_eval(
        split_name=args.forget_split,
        load_in_batch=False,
        batch_size=1,
        prompt_prefix="",
    )
    retain_dataset = dataset.load_dataset_for_eval(
        split_name=args.retain_split,
        load_in_batch=False,
        batch_size=1,
        prompt_prefix="",
    )
    real_authors_dataset = dataset.load_dataset_for_eval(
        split_name=args.holdout_split,
        load_in_batch=False,
        batch_size=1,
        prompt_prefix="",
    )
    world_facts_dataset = dataset.load_dataset_for_eval(
        split_name="world_facts",
        load_in_batch=False,
        batch_size=1,
        prompt_prefix="",
    )

    return {
        "forget": forget_dataset,
        "retain": retain_dataset,
        "real_authors": real_authors_dataset,
        "world_facts": world_facts_dataset,
    }
    
def setup_tofu_perturbed(args):
    from eco.dataset import TOFUPerturbed
    dataset = TOFUPerturbed(None, None)
    forget_dataset = dataset.load_dataset_for_eval(
        split_name=f"{args.forget_split}_perturbed",
        load_in_batch=False,
        batch_size=1,
        prompt_prefix="",
    )
    retain_dataset = dataset.load_dataset_for_eval(
        split_name=f"retain_perturbed",
        load_in_batch=False,
        batch_size=1,
        prompt_prefix="",
    )
    real_authors_dataset = dataset.load_dataset_for_eval(
        split_name=f"{args.holdout_split}_perturbed",
        load_in_batch=False,
        batch_size=1,
        prompt_prefix="",
    )
    world_facts_dataset = dataset.load_dataset_for_eval(
        split_name="world_facts_perturbed",
        load_in_batch=False,
        batch_size=1,
        prompt_prefix="",
    )

    return {
        "forget": forget_dataset,
        "retain": retain_dataset,
        "real_authors": real_authors_dataset,
        "world_facts": world_facts_dataset,
    }

def setup_wandb(args):
    wandb_run = wandb.init(
        project="llm-reload",
        config=args,
    )

    return wandb_run

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    # wandb_run = None
    wandb_run = setup_wandb(args)

    datasets = setup_tofu(args)
    perturbed_datasets = setup_tofu_perturbed(args)

    model = HFModel(
        args.model,
        config_path=f"./config/{args.model_config_dir}",
        generation_config=GenerationConfig(
            do_sample=False, max_new_tokens=512, use_cache=True
        ),
    ) 

    if wandb_run is not None:
        portion_of_retain = args.retain_sample_size / len(datasets["retain"])
        wandb_run.log({
            "portion_of_retain": portion_of_retain,
        })

    _ = evaluate_model_all(model, datasets["forget"], datasets["retain"], datasets["real_authors"], datasets["world_facts"], perturbed_datasets["forget"], perturbed_datasets["retain"], perturbed_datasets["real_authors"], perturbed_datasets["world_facts"], args, wandb_run=wandb_run, eval_handle="base")
    
    model = main_full_batch(model, args, datasets["forget"], datasets["retain"], datasets["real_authors"], datasets["world_facts"], wandb_run=wandb_run)
    model = finetune_model(
        model,
        datasets["retain"],
        args,
        all_params=args.ft_all_params in ["true", "True"],
        wandb_run=wandb_run,
    )
    combination = evaluate_model_all(model, datasets["forget"], datasets["retain"], datasets["real_authors"], datasets["world_facts"], perturbed_datasets["forget"], perturbed_datasets["retain"], perturbed_datasets["real_authors"], perturbed_datasets["world_facts"], args, wandb_run=wandb_run, eval_handle="final")

    if wandb_run is not None:
        wandb_run.log({
            "max_objective_utility_and_quality": combination,
        })
        wandb_run.finish()