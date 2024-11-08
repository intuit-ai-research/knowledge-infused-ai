import argparse
import collections
import json
import subprocess
import os
import re
import string
import sys
import time

import transformers
import torch
import datasets

import numpy as np
from datasets import Dataset

try:
  assert torch.cuda.is_available() is True
except AssertionError:
  print("Please set up a GPU before using LLaMA Factory: https://medium.com/mlearning-ai/training-yolov4-on-google-colab-316f8fff99c6")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(gold_answers, preds):
    exact_scores = []
    f1_scores = []
    for idx, value in enumerate(gold_answers):
        pred = preds[idx]
        exact_score = compute_exact(value, pred)
        f1_score = compute_f1(value, pred)
        exact_scores.append(exact_score)
        f1_scores.append(f1_score)
    return exact_scores, f1_scores


def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map='cuda')
    return tokenizer, model

def inference(tokenizer, model, question):
    with torch.no_grad():
        # Tokenize input question and move to GPU if available
        input_ids = tokenizer.encode(question, return_tensors="pt").to(device)

        # Generate output
        output = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            top_k=50
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    response = generated_text.split("->:")[-1]
    return response

def get_qa(*args):
    if args[0]["role"] == "user":
        question = args[0]["content"]
        answer = args[1]["content"]
    else:
        question = args[1]["content"]
        answer = args[0]["content"]
    return question, answer


def get_predictions(tokenizer, model, dataset, output_path, model_path):
    print("Generate answers")
    predictions = []
    golden_answers = []
    count = 0
    for i in dataset:
        question, answer = get_qa(*i["messages"])
        pred = inference(tokenizer, model, question)
        predictions.append(pred)
        golden_answers.append(answer)
        print(count)
        count += 1
    data = {'predictions': predictions , 'golden_answers': golden_answers}
    dataset = Dataset.from_dict(data)
    dataset.save_to_disk(f"{output_path}/{model_path}")
    return golden_answers, predictions


def evaluate_predictions(golden_answers, predictions, output_path):
    exact_scores, f1_scores = get_raw_scores(golden_answers, predictions)
    exact_match = sum(exact_scores)/len(exact_scores)
    f1 = sum(f1_scores)/len(f1_scores)
    return exact_match, f1


def load_dataset(dataset_path, test_samples):
    dataset = datasets.load_from_disk(dataset_path)
    selected_dataset = dataset.select(range(test_samples))
    return selected_dataset


def train_model(train_args, gpu_args):
    print("Start training")
    # Convert train_args to dictionary and save to JSON file
    json_file = f"{train_args.template}_{train_args.dataset}_{train_args.max_samples}_{train_args.stage}.json"
    train_args_dict = vars(train_args)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(train_args_dict, f, indent=2)
    # Run training command
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = gpu_args.cuda_visible_devices
    subprocess.run(["llamafactory-cli", "train", json_file], env=my_env)


def merge_weights(train_args, merge_args):
    export_dir = f"{train_args.output_dir}_merged"
    print(f"Start merging weights. output_path={export_dir}")
    subprocess.run([
        "llamafactory-cli", "export",
        "--model_name_or_path", train_args.model_name_or_path,
        "--adapter_name_or_path", train_args.output_dir,
        "--template", train_args.template,
        "--finetuning_type", train_args.finetuning_type,
        "--export_dir", export_dir,
        "--export_size", str(merge_args.export_size),
        "--export_device", merge_args.export_device,
        "--export_legacy_format", str(merge_args.export_legacy_format)
    ])

def evaluate_model(train_args, eval_args):
    print("Start evaluation")
    export_dir = f"{train_args.output_dir}_merged"
    test_output_dir = eval_args.test_output_dir
    dataset = load_dataset(eval_args.test_dataset, eval_args.test_samples)
    get_model_metrics(train_args.model_name_or_path, dataset, test_output_dir)
    get_model_metrics(export_dir, dataset, test_output_dir)


def get_model_metrics(model_path, dataset, test_output_dir):
    tokenizer, model = load_model(model_path)
    golden_answers, predictions = get_predictions(tokenizer, model, dataset, test_output_dir, model_path)
    exact_match, f1 = evaluate_predictions(golden_answers, predictions, test_output_dir)
    msg = f"\ncurrent time: {time.time()} model: {model_path}, dataset: {dataset}\nexact_match: {exact_match}\nf1: {f1}"
    print(msg)
    with open(f"{test_output_dir}/metrics.txt", "a") as f:
        f.write(msg)


def get_args_by_group(args, group):
    return argparse.Namespace(**{k: v for k, v in vars(args).items() if k in [action.dest for action in group._group_actions]})


def main():
    parser = argparse.ArgumentParser(description="Fine-tune and merge weights for Mistral model")
    # Training arguments group
    gpu_group = parser.add_argument_group('GPU Arguments')
    gpu_group.add_argument("--cuda_visible_devices", default="0", help="0 for single gpu; 0,1,2,3... for multi-gpu")
    # Training arguments group
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument("--stage", default="sft", help="Training stage")
    train_group.add_argument("--do_train", type=bool, default=True, help="Whether to train")
    train_group.add_argument("--model_name_or_path", default="mistralai/Mistral-7B-Instruct-v0.2", help="Model name or path")
    train_group.add_argument("--dataset", default="fiqa_1gram", help="Dataset name")
    train_group.add_argument("--template", default="mistral", help="Template name")
    train_group.add_argument("--finetuning_type", default="lora", help="Fine-tuning type")
    train_group.add_argument("--lora_target", default="all", help="LoRA target")
    train_group.add_argument("--output_dir", default="mistral_lora_instruct", help="Output directory")
    train_group.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    train_group.add_argument("--lr_scheduler_type", default="cosine", help="Learning rate scheduler type")
    train_group.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    train_group.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    train_group.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    train_group.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    train_group.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs")
    train_group.add_argument("--max_samples", type=int, help="Max samples")
    train_group.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    train_group.add_argument("--quantization_bit", type=int, default=4, help="Quantization bit")
    train_group.add_argument("--loraplus_lr_ratio", type=float, default=16.0, help="LoRA+ learning rate ratio")
    train_group.add_argument("--fp16", type=bool, default=True, help="Use float16 mixed precision training")
    train_group.add_argument("--cache_dir", default="/home/ec2-user/.cache/huggingface/hub", help="cache folder")
    # Merging arguments group
    merge_group = parser.add_argument_group('Merging Arguments')
    merge_group.add_argument("--export_size", type=int, default=2, help="The file shard size (in GB) of the exported model.")
    merge_group.add_argument("--export_device", default="cpu", help="The device used in model export, use cuda to avoid addmm errors.")
    merge_group.add_argument("--export_legacy_format", type=bool, default=False, help="Whether or not to save the `.bin` files instead of `.safetensors`.")

    eval_group = parser.add_argument_group('Eval Arguments')
    eval_group.add_argument("--test_samples", type=int, default=500, help="Number of test samples")
    eval_group.add_argument("--test_dataset", help="Path to test dataset")
    eval_group.add_argument("--test_output_dir", help="Where to store predictions and metrics")


    # Parse arguments
    args = parser.parse_args()
    # Split arguments into their respective groups
    gpu_args = get_args_by_group(args, gpu_group)
    train_args = get_args_by_group(args, train_group)
    merge_args = get_args_by_group(args, merge_group)
    eval_args = get_args_by_group(args, eval_group)
    print(f"gpu_args={gpu_args}\ntrain_args={train_args}\nmerge_args={merge_args}\neval_args={eval_args}")
    # Pass the arguments to the respective functions
    env = os.environ.copy()
    # Now set the CUDA_VISIBLE_DEVICES environment variable
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_args.cuda_visible_devices)
    train_model(train_args, gpu_args)
    merge_weights(train_args, merge_args)
    evaluate_model(train_args, eval_args)


if __name__ == "__main__":
    main()
