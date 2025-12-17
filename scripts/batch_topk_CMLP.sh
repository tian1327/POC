#!/bin/bash

datasets=(
    "semi-aves"
    "species196_insecta"
    "species196_weeds"
    "species196_mollusca"
    "fungitastic-m"
    "fish-vista-m"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "CMLP on $dataset"
    bash scripts/run_dataset_seed_topk_CMLP.sh $dataset 1
    bash scripts/run_dataset_seed_topk_CMLP.sh $dataset 2
    bash scripts/run_dataset_seed_topk_CMLP.sh $dataset 3
done