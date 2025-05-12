import torch
from tqdm import tqdm
from eco.evaluator.utils import answer_prob, log_truth_ratio


class TruthRatio:
    name = "truth_ratio"

    def __init__(self, mode):
        super().__init__()
        assert mode in [
            "min",
            "clip",
        ], f"mode {mode} not supported, use 'min' or 'clip'"
        self.mode = mode

    def evaluate(self, prompts, answers, model, tokenizer):
        truth_ratios = []
        wrapped_prompts_and_answers = tqdm(
            zip(prompts, answers), desc="Evaluating Truth Ratio", unit="prompt", total=len(prompts)
        )
        for prompt, answer in wrapped_prompts_and_answers:
            prompt = [prompt] * len(answer)
            log_probs = answer_prob(prompt, answer, model, tokenizer, "mean")
            log_probs = torch.stack(log_probs).tolist()
            truth_ratios.append(log_truth_ratio(log_probs[1:], log_probs[0]))
        print(f"Log Prob: {log_probs}")
        if self.mode == "min":
            truth_ratios = [min(tr, 1 / tr) for tr in truth_ratios]
        elif self.mode == "clip":
            truth_ratios = [max(0, 1 - tr) for tr in truth_ratios]
        return truth_ratios
