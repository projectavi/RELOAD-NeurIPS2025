import torch
from transformers import GenerationConfig
from scipy.stats import ks_2samp
import numpy as np
from tqdm import tqdm

from eco.model import HFModel
from eco.evaluator import AnswerProb, TruthRatio, ROUGE, ROUGERecall, ChoiceByTopLogit, ChoiceByTopProb

def answer_prob(model, dataset):

    evaluator = AnswerProb(to_prob=True)

    results = evaluator.evaluate(
        prompts=dataset["prompt"],
        answers=dataset["answer"],
        model=model,
        tokenizer=model.tokenizer,
    )

    return sum(results) / len(results)

def truth_ratio(model, dataset, perturbed_dataset, avgd=True):

    evaluator = TruthRatio(mode="clip") # "clip" is the mode discussed in the TOFU paper

    results = evaluator.evaluate(
        prompts=perturbed_dataset["prompt_formatted"],
        answers=perturbed_dataset["choices"],
        model=model,
        tokenizer=model.tokenizer,
    )

    if avgd: 
        return sum(results) / len(results)
    else:
        return results

def rouge_l(model, dataset, args):

    # Generate answers using the model
    generated_answers = []
    wrapped_prompts = tqdm(dataset["prompt"], desc="Generating answers for ROUGE L", unit="prompt")
    for prompt in wrapped_prompts:
        output = model.generate(
            **model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
                    model.device
                ),
                generation_config=model.generation_config,
                eos_token_id=model.tokenizer.eos_token_id,
        )
        generated_answers.append(model.tokenizer.batch_decode(output, skip_special_tokens=False)[0])

    evaluator = ROUGERecall(mode="rougeL")

    results = evaluator.evaluate(
        answers=dataset["answer"],
        generated_answers=generated_answers,
    )

    return sum(results) / len(results)

def forget_quality(model, forget_dataset, forget_perturbed_dataset, args):

    unlearned_truth_ratio = truth_ratio(
        model=model,
        dataset=forget_dataset,
        perturbed_dataset=forget_perturbed_dataset,
        avgd=False
    )

    retrained_model_name = f"{args.model}-{args.retain_split}"

    retrained_model = HFModel(
        retrained_model_name,
        config_path=f"./config/{args.model_config_dir}",
        generation_config=GenerationConfig(
            do_sample=False, max_new_tokens=512, use_cache=True
        ),
    )

    retrained_truth_ratio = truth_ratio(
        model=retrained_model,
        dataset=forget_dataset,
        perturbed_dataset=forget_perturbed_dataset,
        avgd=False
    )

    print("Unlearned Truth Ratio: ", unlearned_truth_ratio)
    print(f"Retrained Truth Ratio: ", retrained_truth_ratio)

    # Kolmogorov-Smirnov test
    result = ks_2samp(np.array(unlearned_truth_ratio), np.array(retrained_truth_ratio))

    return result.pvalue

def model_utility(model, retain_dataset, real_authors_dataset, world_facts_dataset, retain_perturbed_dataset, real_authors_perturbed_dataset, world_facts_perturbed_dataset, args):

    retain_scores = {}

    retain_scores["probability"] = answer_prob(
        model=model,
        dataset=retain_dataset
    )
    retain_scores["truth_ratio"] = truth_ratio(
        model=model,
        dataset=retain_dataset,
        perturbed_dataset=retain_perturbed_dataset
    )
    retain_scores["rouge_l"] = rouge_l(
        model=model,
        dataset=retain_dataset,
        args=args
    )

    real_authors_scores = {}

    real_authors_scores["probability"] = answer_prob(
        model=model,
        dataset=real_authors_dataset
    )
    real_authors_scores["truth_ratio"] = truth_ratio(
        model=model,
        dataset=real_authors_dataset,
        perturbed_dataset=real_authors_perturbed_dataset
    )
    real_authors_scores["rouge_l"] = rouge_l(
        model=model,
        dataset=real_authors_dataset,
        args=args
    )

    world_facts_scores = {}

    world_facts_scores["probability"] = answer_prob(
        model=model,
        dataset=world_facts_dataset
    )
    world_facts_scores["truth_ratio"] = truth_ratio(
        model=model,
        dataset=world_facts_dataset,
        perturbed_dataset=world_facts_perturbed_dataset
    )
    world_facts_scores["rouge_l"] = rouge_l(
        model=model,
        dataset=world_facts_dataset,
        args=args
    )

    # Calculate the harmonic mean of the scores
    harmonic_mean = 9 / (sum(1/(x+1e-20) for x in real_authors_scores.values()) + sum(1/(x+1e-20) for x in world_facts_scores.values()) + sum(1/(x+1e-20) for x in retain_scores.values()))

    return harmonic_mean, retain_scores, real_authors_scores, world_facts_scores

def choice_by_top_logit(model, dataset, args):

    evaluator = ChoiceByTopLogit()

    top_logit_choices = evaluator.evaluate(
        prompts=dataset["prompt"],
        answers=[str(a) for a in dataset["answer"]],
        model=model,
        tokenizer=model.tokenizer,
    )

    print("Top Logit Choices: ", top_logit_choices)

    total_correct = 0
    for i in range(len(top_logit_choices)):
        if top_logit_choices[i] == dataset["answer"][i]:
            total_correct += 1
    accuracy = total_correct / len(dataset["answer"])

    return accuracy

def choice_by_top_prob(model, dataset, args):

    evaluator = ChoiceByTopProb()

    top_prob_choices = evaluator.evaluate(
        prompts=dataset["prompt"],
        answers=dataset["choices"],
        model=model,
        tokenizer=model.tokenizer,
    )

    print("Top Prob Choices: ", top_prob_choices)

    total_correct = 0
    for i in range(len(top_prob_choices)):
        if top_prob_choices[i] == dataset["answer"][i]:
            total_correct += 1

    accuracy = total_correct / len(dataset["answer"])

    return accuracy
