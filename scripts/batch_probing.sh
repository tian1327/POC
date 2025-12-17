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
    echo "Few-shot linear probing on $dataset"
    bash scripts/run_dataset_seed_probing.sh $dataset 1
    bash scripts/run_dataset_seed_probing.sh $dataset 2
    bash scripts/run_dataset_seed_probing.sh $dataset 3

done