#!/bin/bash

datasets=(
    # "semi-aves"
    # "species196_insecta"
    # "species196_weeds"
    "species196_mollusca"
    # "fungitastic-m"
    # "fish-vista-m"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "fewshot finetuning on $dataset"
    bash scripts/run_dataset_seed_topk_fewshot_finetune.sh $dataset 1
    # bash scripts/run_dataset_seed_topk_fewshot_finetune.sh $dataset 2
    # bash scripts/run_dataset_seed_topk_fewshot_finetune.sh $dataset 3
done