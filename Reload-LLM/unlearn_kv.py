# MAIN FILE FOR RELOAD UNLEARNING ON LLMS
import torch
from accelerate import Accelerator
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import GenerationConfig, pipeline, default_data_collator
from itertools import chain

from eco.model import HFModel
from cft import send_requests, parse_responses
import time

import pandas as pd

from eco.utils import seed_everything

per_device_batch_size = 1

"""
This pipeline uses a QA structure in order to extract the knowledge the LLM has about a specific topic and then uses CausalLM in order to remove it during some finetuning steps.
"""

# class MyDataset(Dataset):
#     def __init__(self, num_samples):
#         super().__init__()
#         self.len = num_samples

#     def __getitem__(self, index):
#         input_ids = torch.arange(1, index + 2, dtype=torch.float32)
#         labels = torch.remainder(input_ids, 2)
#         return {"input_ids": input_ids, "labels": labels}

#     def __len__(self):
#         return self.len
    
import torch

device = "cuda:0"

class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def collate_fn(features):
    print(features)
    # input_ids = torch.nn.utils.rnn.pad_sequence([f["input_ids"].unsqueeze(0).to(device) for f in features], batch_first=True,
                                                # padding_value=-100)
    # labels = torch.nn.utils.rnn.pad_sequence([f["input_ids"].unsqueeze(0).to(device) for f in features], batch_first=True, padding_value=-100)
    input_ids = torch.tensor([f["input_ids"].to(device) for f in features]).to(device)
    labels = torch.tensor([f["input_ids"].to(device) for f in features]).to(device)
    return {"input_ids": input_ids[..., None], "labels": labels[..., None]}

seed_everything(0)
model_name = "phi-2"

# STEP 1: Load the model

model = HFModel(
    model_name,
    config_path="./config/model_config",
    generation_config=GenerationConfig(
        do_sample=False, max_new_tokens=256, use_cache=True
    ),
)

selected_params = []
for name, param in model.model.named_parameters():
    if "attn" in name:
        selected_params.append(name)
print(f"Selected {len(selected_params)} parameters for unlearning") 

# STEP 2: Load the tokenizer

tokenizer = model.tokenizer

block_size = tokenizer.model_max_length

def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

# STEP 3: Define the forget dataset of prompts

# forget_prompts = ["Who is Harry Potter?"]
forget_prompts = ["Who is Luke Skywalker?"]
original_prompt = forget_prompts[0] # Assuming only one prompt

# STEP 4: Generate the forget dataset of responses

prompt = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": original_prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

# generator = pipeline("text-generation", model=model.model, tokenizer=model.tokenizer)
# print(generator(forget_prompts[0]))
# exit(0)
forget_output = model.generate(
    **model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
            model.device
        ),
        generation_config=model.generation_config,
        eos_token_id=model.tokenizer.eos_token_id,
)

# STEP 5: Add in CFT to the forget dataset of prompts

output_text = model.tokenizer.batch_decode(forget_output, skip_special_tokens=False)[0][len(prompt) :]
print("Original output:")
print(output_text)
batch_ids = send_requests([prompt], [output_text])
# print(batch_ids)
# time.sleep(5)
results = parse_responses(batch_ids)

# print("contextualise")
# print(results[0]["response"]) # Assuming there is a single entry in here

contextualised_prompt = results[0]["response"].split("Contextual Prompt: ")[1]

full_prompt = contextualised_prompt + original_prompt

prompt = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": full_prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

extracted_knowledge = model.generate(
    **model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
            model.device
        ),
        generation_config=model.generation_config,
        eos_token_id=model.tokenizer.eos_token_id,
)

output_text = model.tokenizer.batch_decode(extracted_knowledge, skip_special_tokens=False)[0][len(prompt) :]
print("Extracted output:")
print(output_text)

completed_prompt = original_prompt + output_text

# print("Cleaned prompt")
# print(contextualised_prompt + original_prompt)

# contextualised_prompt = contextualised_prompt + " Luke Skywalker is"

# # Inference step to obtain the full description of the LLM's knowledge
# generator = pipeline("text-generation", model=model.model, tokenizer=model.tokenizer)
# print("Extracted knowledge")
# print(generator(contextualised_prompt)[0]["generated_text"])
# exit(0)

# exit(0)

# STEP 6: Pass over target knowledge extracted with prompt and embedded with CFT, and collect backprop gradients accumulated

embedding_size = model.model.get_input_embeddings().weight.shape[0]
if len(model.tokenizer) > embedding_size:
    model.model.resize_token_embeddings(len(model.tokenizer))

train_df = pd.DataFrame([completed_prompt])

# tokenized_dataset = model.tokenizer(output_text, return_tensors="pt").to(device)
dataset = Dataset.from_pandas(train_df.rename(columns={0: "train"}), split="train")
tokenized_dataset = dataset.map(lambda samples: tokenizer(samples["train"]), batched=True)
# print(tokenized_dataset)
lm_datasets = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=1,
            desc=f"Grouping texts in chunks of {block_size}",
        )
# print(lm_datasets)
train_dataset = lm_datasets["train"]
# print(train_dataset)
train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=1
    )

# print(train_dataloader)

# for step, batch in enumerate(train_dataloader):
#     # print(batch)
#     outputs = model.model(**batch)
#     loss = outputs.loss
#     print(outputs)
    # exit(0)

model.model.train()

inputs = model.tokenizer(completed_prompt, return_tensors="pt").to(device)
# inputs["labels"] = inputs["input_ids"].copy_()

# print(inputs)
# exit(0)

outputs = model.model.generate(**inputs)
# print(outputs.shape)

output_list = [outputs.to(device)]

# print(outputs.shape)
# exit(0)

# dataset = Dataset.from_dict({"inputs": forget_prompts, "labels": output_list})
dataset = Dataset.from_dict({"input_ids": [t.to(device) for t in inputs["input_ids"]], "labels": [o.to(device) for o in output_list]})
dataloader = DataLoader(
    dataset, shuffle=False, collate_fn=default_data_collator, batch_size=1
)
# print(dataset)
# print(dataloader)

# accelerator = Accelerator(gradient_accumulation_steps=1)
# dataset = MyDataset(tokenizer(output_text))
# # dataloader = DataLoader(dataset, batch_size=per_device_batch_size, collate_fn=collate_fn)
# dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
# criterion = torch.nn.CrossEntropyLoss(reduction="sum") # must sum over samples rather than averaging
# model_optimizer = torch.optim.SGD(model.model.parameters(), lr=0.08)

# model, model_optimizer = accelerator.prepare(model, model_optimizer)
# dataloader = accelerator.prepare_data_loader(dataloader, device_placement=True)
# training_iterator = iter(dataloader)
#
# num_samples_in_epoch = len(dataloader)
# remainder = num_samples_in_epoch % gradient_accumulation_steps
# remainder = remainder if remainder != 0 else gradient_accumulation_steps
# total_gradient_updates = math.ceil(num_samples_in_epoch)

full_batch_without_accum = next(iter(dataloader))
# print(full_batch_without_accum)
total_inputs, total_labels = full_batch_without_accum["input_ids"].to(device), full_batch_without_accum["labels"].to(device)
print(total_inputs.shape)
print(total_labels.shape)
# total_inputs = {"input_ids": torch.cat((total_inputs, total_labels[0]), dim=1), "labels": torch.cat((total_inputs, total_labels[0]), dim=1)}
total_inputs = {"input_ids": total_labels[0], "labels": total_labels[0]}
print(total_inputs["labels"].shape)

print(f"Total parameters for optimization: {len([param for name, param in model.model.named_parameters() if name in selected_params])}")
optimizer = torch.optim.SGD([param for name, param in model.model.named_parameters() if name in selected_params], lr=0.08)

# train the cloned model
# print(type(model(total_inputs))) # https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast
# loss = torch.nn.CrossEntropyLoss(reduction="mean")(model(total_inputs).logits.view(-1, 2),
                                                #    total_labels.view(-1).to(torch.int64))

model.model.train()
import gc
gc.collect()
torch.cuda.empty_cache() 
output = model(**total_inputs)
loss = output.loss
optimizer.zero_grad()
loss.backward()
forget_gradients = {}

for name, param in model.model.named_parameters():
    if name in selected_params:
        if param.grad is not None:
            forget_gradients[name] = param.grad.clone()

print(f"Keys in forget gradients: {len(forget_gradients.keys())}")

model.model.eval()

# optimizer.zero_grad()

# Ascent step

for name, param in model.model.named_parameters():
    if name in forget_gradients and name in selected_params:
        param.data = param.data + 0.08 * forget_gradients[name]


all_gradients = [tensor.flatten().to("cpu") for tensor in forget_gradients.values()]
# del forget_gradients
all_gradients = torch.cat(all_gradients).to("cpu")
# print(all_gradients.shape)
proportion = 0.001
threshold = torch.max(torch.topk(all_gradients, int(proportion * len(all_gradients)), largest=False).values)
print(threshold)
del all_gradients

for name, param in model.model.named_parameters():
    if name in forget_gradients and name in selected_params:
        print(f"Resetting {forget_gradients[name][forget_gradients[name] < threshold].numel()} parameters in layer {name}")
        param.data = torch.where(forget_gradients[name] < threshold, torch.mean(param.data), param.data)
        
# prompt = model.tokenizer.apply_chat_template(
#     [{"role": "user", "content": prompt }],
#     tokenize=False,
#     add_generation_prompt=True,
# )

print(f"Prompt: {prompt}")

forget_output = model.generate(
    **model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
            model.device
        ),
        generation_config=model.generation_config,
        eos_token_id=model.tokenizer.eos_token_id,
)

print("After ascent and reset step:")
output_text = model.tokenizer.batch_decode(forget_output, skip_special_tokens=False)[0][len(prompt) :]
print(output_text)

del forget_output
del prompt
del output_text

print("Testing base prompt")

test_prompt = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Who is Luke Skywalker?"}],
    tokenize=False,
    add_generation_prompt=True,
)

test_output = model.generate(
    **model.tokenizer(test_prompt, add_special_tokens=False, return_tensors="pt").to(
            model.device
        ),
        generation_config=model.generation_config,
        eos_token_id=model.tokenizer.eos_token_id,
)

output_text = model.tokenizer.batch_decode(test_output, skip_special_tokens=False)[0]
print(output_text)

print("Testing other knowledge")

test_prompt = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Who is Harry Potter?"}],
    tokenize=False,
    add_generation_prompt=True,
)

test_output = model.generate(
    **model.tokenizer(test_prompt, add_special_tokens=False, return_tensors="pt").to(
            model.device
        ),
        generation_config=model.generation_config,
        eos_token_id=model.tokenizer.eos_token_id,
)

output_text = model.tokenizer.batch_decode(test_output, skip_special_tokens=False)[0]
print(output_text)

# THIS IS ALL FOR THE FUTURE - lets test this computation and an ascent step
# STEP 7: Define the retain dataset of prompts

# STEP 8: Add in CFT to the retain dataset of prompts

# STEP 9: Generate the retain dataset of responses

# STEP 10: Save gradients

# STEP 11: Calculate Knowledge-Values for attention parameters

# STEP 12: Reset the top-k attention parameters

# STEP 13: Train CFT on the retain dataset