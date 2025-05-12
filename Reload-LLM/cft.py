# This file will define the CFT collection methods to generate Curriculum-Coded prompts

# CODE ADAPTED FROM ETHAN FOR GENERATING CFT FROM CLAUDE

import anthropic
from datasets import load_dataset
import json
import time
import os
import typing

def load_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_responses(batch_ids, client, batch_custom_map=None, prev_responses=None):
    all_responses = []
    if batch_custom_map is None:
        batch_custom_map = {}
    for batch_id in batch_ids:
        if batch_id not in batch_custom_map:
            batch_custom_map[batch_id] = []
            # print(batch_id)
            # print(client.messages.batches.retrieve(batch_id))
            for result in client.messages.batches.results(batch_id):
                custom_id = batch_id+result.custom_id
                response = result.result.message.content[0].text
                all_responses.append({'custom_id': custom_id, 'response': response})
                batch_custom_map[batch_id].append(custom_id)
        else:
            for custom_id in batch_custom_map[batch_id]:
                for prev_response in prev_responses:
                    if custom_id == prev_response['custom_id']:
                        response = prev_response['response']
                        all_responses.append({'custom_id': custom_id, 'response': response})
                        break

    return all_responses, batch_custom_map


def group_responses(responses, group_size=3):
    grouped = []
    for i in range(0, len(responses), group_size):
        group = responses[i:i + group_size]
        combined_response = ' '.join(item['response'] for item in group)
        grouped.append({'custom_id': f'group-{i // group_size}', 'response': combined_response})
    return grouped


def format_prompt_for_generating_contextual_prompt(content):
    prompt = f"""
Based on the TARGET CONCEPT:

Generate a concise "contextual prompt" that will enhance learning effectiveness and draw out all relevant knowledge. The prompt should:

1. Follow the style of [select one learning theory approach: In-Depth Exploration/Reflective Thinking/Summarization and Synthesis/Focus on Key Concepts/Contextual Understanding/Critical Analysis/Question-Based Learning]

2. Explicitly identify:
   • The fundamental concepts that must be understood
   • Key relationships between important elements
   • Critical facts that require focus for mastery
   • How these elements connect to and are relevant for reasoning or application

3. Be formatted as a directive that encourages active engagement with the material (approximately 3-5 sentences)

4. Frame the learning in a way that facilitates long-term retention, practical application, and maximizes extracting knowledge from the learner.

TARGET CONCEPT: {content}

Your contextual prompt should help the learner not just memorize information but develop a deeper, more applicable understanding of the concept.
"""
    return prompt


def generate_responses_for_contextual_prompts(target_concept, client, batch_size=1, request_api=False):
    BATCH_SIZE = batch_size
    batch_requests = []
    sanity_check = []

    for i in range(0, len(target_concept), BATCH_SIZE):
        batch_content = target_concept[i:i + BATCH_SIZE]
        requests = []

        for idx, item in enumerate(batch_content):
            prompt = format_prompt_for_generating_contextual_prompt(
                item
            )

            requests.append({
                "custom_id": f"prompt-{i + idx}",
                "params": {
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                },
            })

        sanity_check.append(requests)
        if request_api:
            message_batch = client.messages.batches.create(requests=requests)
            batch_requests.append(message_batch)
    print(batch_requests)
    return sanity_check, batch_requests

def load_cft_from_cache(prompts):
    BATCH_SIZE = 1

    if os.path.exists("./request_cache.json"):
        with open("./request_cache.json", "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    flattened_ids = []

    uncached_prompts = []

    for i in range(0, len(prompts), BATCH_SIZE):
        key = ';'.join(prompts[i:i + BATCH_SIZE])
        if key not in cache:
            print(f"ADDING {prompts[i:i + BATCH_SIZE]} to gathering")
            uncached_prompts.extend(prompts[i:i + BATCH_SIZE])
        else:
            flattened_ids.append(cache[key])

    return flattened_ids, uncached_prompts

def send_requests(prompts, concepts):
    client = anthropic.Anthropic(
        api_key='<YOUR-KEY-HERE>',
    )

    print(len(prompts))
    print(len(concepts))
    # exit(0)

    BATCH_SIZE = 1

    if os.path.exists("./request_cache.json"):
        with open("./request_cache.json", "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    request_concepts = []
    flattened_ids = []

    for i in range(0, len(prompts), BATCH_SIZE):
        key = ';'.join(prompts[i:i + BATCH_SIZE])
        if key not in cache:
            print(f"ADDING {prompts[i:i + BATCH_SIZE]} to gathering")
            if isinstance(concepts[i:i + BATCH_SIZE], str):
                request_concepts.append(concepts[i:i + BATCH_SIZE])
            else:
                request_concepts.extend(concepts[i:i + BATCH_SIZE])
            flattened_ids.append(f"KEY:<THISISARELOADCACHETAG>:{key}")
        else:
            flattened_ids.append(cache[key])
    
    print(flattened_ids)
    print(request_concepts)

    if len(request_concepts) > 0:
        requests, batches = generate_responses_for_contextual_prompts(request_concepts, client, batch_size=BATCH_SIZE, request_api=True)
        flattened_requests = [request for batch in requests for request in batch]

        # Save the requests to a file
        with open('contextual_prompt_requests.json', 'w') as f:
            json.dump(flattened_requests, f, indent=2)

        print(f"NUM BATCHES: {len(batches)}")

    j = 0
    for i in range(len(flattened_ids)):
        if flattened_ids[i].startswith("KEY"):
            key = flattened_ids[i].split(":<THISISARELOADCACHETAG>:")[1]
            print(f"J:{j}")
            flattened_ids[i] = batches[j].id
            cache[key] = batches[j].id
            j += 1

    print(flattened_ids)

    with open("./request_cache.json", "w") as f:
        json.dump(cache, f, indent=2)

    return flattened_ids

def parse_responses(batch_ids):

    client = anthropic.Anthropic(
        api_key='<YOUR-KEY-HERE>',
    )

    if os.path.exists("batch_custom_map.json"):
        with open("batch_custom_map.json", "r") as f:
            batch_custom_map = json.load(f)
    else:
        batch_custom_map = None

    if os.path.exists("contextual_prompt_for_grouped_responses.json"):
        with open('contextual_prompt_for_grouped_responses.json', 'r') as f:
            prev_responses = json.load(f)
    else:
        prev_responses = []

    all_responses, batch_custom_map = get_responses(batch_ids, client, batch_custom_map, prev_responses)
    with open('contextual_prompt_for_grouped_responses.json', 'w') as f:
        for response in all_responses:
            found = False
            for prev_response in prev_responses:
                if response['custom_id'] == prev_response['custom_id']:
                    found = True
                    break
            if not found:
                prev_responses.append(response)
        json.dump(prev_responses, f, indent=2)

    with open('batch_custom_map.json', 'w') as f:
        json.dump(batch_custom_map, f, indent=2)

    return all_responses


if __name__ == "__main__":
    # educational_content = load_json_file('educational_content_responses.json')
    # educational_content = group_responses(educational_content)
    # with open('grouped_educational_content.json', 'w') as f:
    #     json.dump(educational_content, f, indent=2)
    educational_content = load_json_file('grouped_educational_content.json')

    client = anthropic.Anthropic(
        api_key="shouldnotwork"
    )

    # requests = generate_responses_for_contextual_prompts(educational_content, request_api=False)
    # flattened_requests = [request for batch in requests for request in batch]
    import pdb;

    pdb.set_trace()

    batch_ids = ["msgbatch_01DNE5Xxwyc9iHgPmGF63YTi",
                 "msgbatch_01HG2thmd4CNTExsAGLtfW9v",
                 "msgbatch_01TwQm3pEENTANvfxrJWy6Ps",
                 "msgbatch_01Fs1P3feaj2crZpvaaHdgfG",
                 "msgbatch_01VKuLiaXBLU6bCeSXKxUWig",
                 "msgbatch_013kyfrxAfJ5GkURET7HhJdc",
                 "msgbatch_01DP5VPSXyq6DvNouCPsrPmj",
                 "msgbatch_014kYeyNxMGsNG4WowMorpA6",
                 "msgbatch_01MMsh2ukZffHq2xkT9pNptW"]

    all_responses = get_responses(batch_ids)
    with open('contextual_prompt_for_grouped_responses.json', 'w') as f:
        json.dump(all_responses, f, indent=2)
