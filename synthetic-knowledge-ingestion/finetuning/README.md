# SFT and CPT pipeline

This pipeline is created based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a comprehensive framework that integrates advanced efficient training techniques on various LLMs. It allows us to first run the training script, and then merge the model weights and get evaluation metrics on test data.

## Folder schema

```sh
- data/ # folder to store the datasets
- LLaMA-Factory/ # will be cloned when running requirements_ft.sh
- requirements_ft.sh # install pkgs needed for finetuning
- requirements_data.sh # install pkgs needed for creating dataset
- data_preparation.py # python script for creating dataset
- ft_llama_factory.py # python script for running fine-tuning job using llama-factory
```

## How to run this fine-tuning pipeline

### SFT

Set stage to `sft` and have data in the following format:

```json
{
    "messages": [
        {
            "role": "user",
            "content": {{question}}
        },
        {
            "role": "assistant",
            "content": {{answer}}
        }
    ]
}
```

### CPT

Set stage to `pt` and have data in the following format:

```json
{"text": f"question: {{question}}\nanswer: {{answer}}"}
```

### Sample commands to run

```sh
# create data
source ./requirements_data.sh
python data_preparation.py --llama_factory_path ./LLaMA-Factory --raw_data_path ./data/fiqa/fiqa_1.json --output_path ./data/fiqa_1_gram
python data_preparation.py --llama_factory_path ./LLaMA-Factory --raw_data_path ./data/fiqa/fiqa_2.json --output_path ./data/fiqa_2_gram
python data_preparation.py --llama_factory_path ./LLaMA-Factory --raw_data_path ./data/fiqa/fiqa_3.json --output_path ./data/fiqa_3_gram

python data_preparation.py --llama_factory_path ./LLaMA-Factory --raw_data_path ./data/fiqa/fiqa_1.json --output_path ./data/fiqa_1_gram --task pt
python data_preparation.py --llama_factory_path ./LLaMA-Factory --raw_data_path ./data/fiqa/fiqa_2.json --output_path ./data/fiqa_2_gram --task pt
python data_preparation.py --llama_factory_path ./LLaMA-Factory --raw_data_path ./data/fiqa/fiqa_3.json --output_path ./data/fiqa_3_gram --task pt

python data_preparation.py --llama_factory_path ./LLaMA-Factory --raw_data_path ./data/fiqa_1_gram --output_path ./data/fiqa_1_gram --transform True

python data_preparation.py --llama_factory_path ./LLaMA-Factory --raw_data_path ./data/fiqa/fiqa_1.json ./data/fiqa/fiqa_2.json ./data/fiqa/fiqa_3.json --output_path ./data/fiqa_all_gram --train_test_ratio 0.1 --train_val_ratio 0

python data_preparation.py --llama_factory_path ./LLaMA-Factory --raw_data_path ./data/fiqa_all_gram --output_path ./data/fiqa_all_gram --transform --task pt

 # install required packages
source ./requirements_ft.sh
# move finetune code to LLaMA-Factory folder
mv ../ft_llama_factory.py .
# create a new folder in LLaMA-Factory to store the experiments
mkdir -p experiments/llama2 experiments/mistral experiments/gemma experiments/phi3
# finetune

# google/gemma-2b
#sft
python ft_llama_factory.py --stage sft \
    --model_name_or_path google/gemma-2b \
    --dataset fiqa_2_gram_ft \
    --template gemma \
    --max_samples 50 \
    --output_dir gemma_lora \
    --test_samples 5 \
    --test_dataset ../data/fiqa_2_gram_ft/test \
    --test_output_dir phi3_lora_test_result

#sft
python ft_llama_factory.py --stage sft \
    --model_name_or_path google/gemma-2b \
    --dataset fiqa_1_gram_ft \
    --template gemma \
    --max_samples 5000 \
    --output_dir gemma_lora \
    --test_samples 500 \
    --test_dataset ../data/fiqa_1_gram_ft/test \
    --test_output_dir gemma_lora_test_result \

# sft using all gram data
nohup python ft_llama_factory.py --stage sft \
    --model_name_or_path google/gemma-2b \
    --dataset fiqa_all_gram_sft \
    --template gemma \
    --output_dir experiments/gemma/gemma_lora_sft_fiqa_all_gram \
    --test_samples 500 \
    --test_dataset ../data/fiqa_all_gram_sft/test \
    --test_output_dir experiments/gemma/gemma_lora_sft_fiqa_all_gram_test_result \
    --cuda_visible_devices 0,1,2,3 \
    > experiments/gemma/log_gemma_lora_sft_fiqa_all_gram.log 2>&1

# pt using 1 gram data
nohup python ft_llama_factory.py --stage pt \
    --model_name_or_path google/gemma-2b \
    --dataset fiqa_1_gram_pt \
    --template gemma \
    --max_samples 5000 \
    --output_dir experiments/gemma/gemma_lora_pt \
    --test_samples 500 \
    --test_dataset ../data/fiqa_1_gram_ft/test \
    --test_output_dir experiments/gemma/gemma_lora_pt_test_result \
    > experiments/gemma/log_gemma_lora_pt.log 2>&1

# pt using all data
nohup python ft_llama_factory.py --stage pt \
    --model_name_or_path google/gemma-2b \
    --dataset fiqa_all_gram_pt \
    --template gemma \
    --output_dir experiments/gemma/gemma_lora_pt_fiqa_all_gram \
    --test_samples 500 \
    --test_dataset ../data/fiqa_all_gram_ft/test \
    --test_output_dir experiments/gemma/gemma_lora_pt_fiqa_all_gram_test_result \
    > experiments/gemma/log_gemma_lora_pt_fiqa_all_gram.log 2>&1

# phi-3
python ft_llama_factory.py --stage sft \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --dataset fiqa_2_gram_ft \
    --template phi \
    --max_samples 50 \
    --output_dir phi3_lora \
    --test_samples 5 \
    --test_dataset ../data/fiqa_2_gram_ft/test \
    --test_output_dir phi3_lora_test_result

# sft using all data
nohup python ft_llama_factory.py --stage sft \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --dataset fiqa_all_gram_sft \
    --template phi \
    --output_dir experiments/phi3/phi3_lora_sft_fiqa_all_gram \
    --test_samples 500 \
    --test_dataset ../data/fiqa_all_gram_ft/test \
    --test_output_dir experiments/phi3/phi3_lora_sft_fiqa_all_gram_test_result \
    > experiments/phi3/log_phi3_lora_sft_fiqa_all_gram.log 2>&1

# pt using 1 gram data
nohup python ft_llama_factory.py --stage pt \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --dataset fiqa_1_gram_pt \
    --template phi \
    --max_samples 5000 \
    --output_dir experiments/phi3/phi3_lora_pt \
    --test_samples 500 \
    --test_dataset ../data/fiqa_1_gram_ft/test \
    --test_output_dir experiments/phi3/phi3_lora_pt_test_result \
    > experiments/phi3/log_phi3_lora_pt.log 2>&1

# pt using all data
nohup python ft_llama_factory.py --stage pt \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --dataset fiqa_all_gram_pt \
    --template phi \
    --output_dir experiments/phi3/phi3_lora_pt_fiqa_all_gram \
    --test_samples 500 \
    --test_dataset ../data/fiqa_all_gram_ft/test \
    --test_output_dir experiments/phi3/phi3_lora_pt_fiqa_all_gram_test_result \
    > experiments/phi3/log_phi3_lora_pt_fiqa_all_gram.log 2>&1

# mistral
python ft_llama_factory.py --stage sft \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset fiqa_2_gram_ft \
    --template mistral \
    --max_samples 50 \
    --output_dir mistral_lora \
    --test_samples 5 \
    --test_dataset ../data/fiqa_2_gram_ft/test \
    --test_output_dir mistral_lora_test_result

# pt using 1 gram data
nohup python ft_llama_factory.py --stage pt \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset fiqa_1_gram_pt \
    --template mistral \
    --max_samples 5000 \
    --output_dir experiments/mistral/mistral_lora_pt \
    --test_samples 500 \
    --test_dataset ../data/fiqa_1_gram_ft/test \
    --test_output_dir experiments/mistral/mistral_lora_pt_test_result \
    > experiments/mistral/log_mistral_lora_pt.log 2>&1

# pt using all data
nohup python ft_llama_factory.py --stage pt \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset fiqa_all_gram_pt \
    --template mistral \
    --output_dir experiments/mistral/mistral_lora_pt \
    --test_samples 500 \
    --test_dataset ../data/fiqa_all_gram_ft/test \
    --test_output_dir experiments/mistral/mistral_lora_pt_fiqa_all_gram_test_result \
    > experiments/mistral/log_mistral_lora_pt_fiqa_all_gram.log 2>&1

# llama2
# sft
python ft_llama_factory.py --stage sft \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --dataset fiqa_2_gram_ft \
    --template llama2 \
    --max_samples 50 \
    --output_dir llama2_lora \
    --test_samples 5 \
    --test_dataset ../data/fiqa_2_gram_ft/test \
    --test_output_dir llama2_lora_test_result

# pt
nohup python ft_llama_factory.py --stage pt \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --dataset fiqa_1_gram_pt \
    --template llama2 \
    --max_samples 5000 \
    --output_dir experiments/llama2/llama2_lora_pt \
    --test_samples 500 \
    --test_dataset ../data/fiqa_1_gram_ft/test \
    --test_output_dir experiments/llama2/llama2_lora_pt_test_result \
    > experiments/llama2/log_llama2_lora_pt.log 2>&1

# pt using all gram data
nohup python ft_llama_factory.py --stage pt \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --dataset fiqa_all_gram_pt \
    --template llama2 \
    --output_dir experiments/llama2/llama2_lora_pt_fiqa_all_gram \
    --test_samples 500 \
    --test_dataset ../data/fiqa_all_gram_ft/test \
    --test_output_dir experiments/llama2/llama2_lora_pt_fiqa_all_gram_test_result \
    > experiments/llama2/log_llama2_lora_pt_fiqa_all_gram.log 2>&1
```
