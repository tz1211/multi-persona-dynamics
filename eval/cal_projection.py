import os
import asyncio
import yaml
from typing import Dict, List
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]

def save_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

def cos_sim(a, b):
    return (a*b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))

def a_proj_b(a, b):
    return (a * b).sum(dim=-1) / b.norm(dim=-1)

def main(file_path, vector_path_list=[], layer_list=[], projection_type="proj", model_name="unsloth/Qwen3-4B-Instruct-2507", base_model=None, overwrite=False):
    """
    Calculate projection of model hidden states onto persona vectors.
    
    If base_model is provided, calculates: projection((finetuned_rep - base_rep) onto persona_vector)
    Otherwise, calculates: projection(model_rep onto persona_vector) (original behavior)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    metric_model_name = os.path.basename(model_name) + "_"
    if not isinstance(vector_path_list, list):
        vector_path_list = [vector_path_list]
    if not isinstance(layer_list, list):
        layer_list = [layer_list]
    vector_dict = {}
    layer_dict = {}
    for vector_path in vector_path_list:
        vector = torch.load(vector_path, weights_only=False)
        for layer in layer_list:
            vector_name = vector_path.split("/")[-1].split(".")[0]
            suffix = "_finetuning_shift" if base_model is not None else ""
            metric_name = f"{metric_model_name}{vector_name}_{projection_type}_layer{layer}{suffix}"
            vector_dict[metric_name] = vector[layer]
            layer_dict[metric_name] = layer

    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
        prompts = [d["prompt"] for _, d in data.iterrows()]
        answers = [d["answer"] for _, d in data.iterrows()]
        vector_dict_keys = list(vector_dict.keys())
        for metric_name in vector_dict_keys:
            if metric_name in data.columns:
                if overwrite:
                    print(f"Overwriting {metric_name}")
                else:
                    print(f"Metric {metric_name} already exists, skipping...")
                    vector_dict.pop(metric_name)
    else:
        data = load_jsonl(file_path)
        prompts =  [tokenizer.apply_chat_template(d["messages"][:-1], tokenize=False, add_generation_prompt=True) for d in data]
        answers = [d["messages"][-1]["content"] for d in data]
        vector_dict_keys = list(vector_dict.keys())
        for metric_name in vector_dict_keys:
            if metric_name in data[0].keys():
                if overwrite:
                    print(f"Overwriting {metric_name}")
                else:
                    print(f"Metric {metric_name} already exists, skipping...")
                    vector_dict.pop(metric_name)
    if len(vector_dict) == 0:
        print("No metrics to calculate, exiting...")
        return
    else:
        print(f"Calculating {len(vector_dict)} metrics:")
        for metric_name in vector_dict.keys():
            print(f"{metric_name}")

    # Load main model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    
    # Load base model if provided
    base_model_obj = None
    if base_model is not None:
        print(f"Loading base model: {base_model}")
        base_model_obj = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        base_tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    projections = {k:[] for k in vector_dict.keys()}
    for prompt, answer in tqdm(zip(prompts, answers), total=len(prompts), desc=f"Calculating"):
        inputs = tokenizer(prompt+ answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        outputs = model(**inputs, output_hidden_states=True)
        
        # Get base model hidden states if base_model is provided
        base_outputs = None
        if base_model_obj is not None:
            base_inputs = base_tokenizer(prompt+ answer, return_tensors="pt", add_special_tokens=False).to(base_model_obj.device)
            base_outputs = base_model_obj(**base_inputs, output_hidden_states=True)
        
        for metric_name in vector_dict.keys():
            layer = layer_dict[metric_name]
            vector = vector_dict[metric_name]
            
            # Get finetuned model representation
            response_avg = outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu()
            last_prompt = outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu()
            
            # If base_model is provided, compute difference
            if base_model_obj is not None:
                base_response_avg = base_outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu()
                base_last_prompt = base_outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu()
                
                # Compute difference: finetuned - base
                response_avg = response_avg - base_response_avg
                last_prompt = last_prompt - base_last_prompt
            
            # Project onto persona vector
            if projection_type == "proj":
                projection = a_proj_b(response_avg, vector).item()
            elif projection_type == "prompt_last_proj":
                projection = a_proj_b(last_prompt, vector).item()
            else:
                projection = cos_sim(response_avg, vector).item()
            projections[metric_name].append(projection)
    

    if file_path.endswith(".csv"):
        for metric_name in vector_dict.keys():
            data[metric_name] = projections[metric_name]
        data.to_csv(file_path, index=False)
    else:
        for i, d in enumerate(data):
            for metric_name in vector_dict.keys():
                d[metric_name] = projections[metric_name][i]
        save_jsonl(data, file_path)

    print(f"Projection results saved to {file_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--vector_path_list", type=str, nargs="+", default=[])
    parser.add_argument("--layer_list", type=int, nargs="+", default=[])
    parser.add_argument("--projection_type", type=str, default="proj", choices=["proj", "prompt_last_proj", "cos_sim"])
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base_model", type=str, default=None, help="Base model to compute difference with. If provided, calculates projection of (finetuned_rep - base_rep) onto persona vector.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.file_path, args.vector_path_list, args.layer_list, args.projection_type, args.model_name, args.base_model, args.overwrite)