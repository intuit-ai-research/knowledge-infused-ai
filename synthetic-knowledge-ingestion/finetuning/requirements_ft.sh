#!/bin/bash

export HUGGINGFACE_TOKEN=

python -m venv venv
source venv/bin/activate

rm -rf LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
echo `ls`
pip install -e .[torch,metrics,deepspeed,bitsandbytes,vllm]
pip install accelerate datasets

pip install "huggingface_hub[cli]"
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

mv ./src/fr_llama_factory.py

# # install unsloth
# pip install "unsloth[cu118-torch230] @ git+https://github.com/unslothai/unsloth.git"
# pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
# pip install "unsloth[cu118-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
# pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"