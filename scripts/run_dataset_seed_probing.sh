#!/bin/bash

prefix="LinearProbing"
# prefix="test_LinearProbing"


# methods=("mixup" "saliencymix" "CMO" "cutmix-fs" "resizemix" "CMLP" "probing" "finetune" "FLYP" "cutmix" "fixmatch")
methods=("probing")

# data_sources=("fewshot" "retrieved" "fewshot+retrieved" "fewshot+unlabeled" "fewshot+retrieved+unlabeled")
data_sources=("fewshot")

# shot_values=(16 8 4)
# shot_values=(16)
shot_values=(4)

batch_size=32

loss="CE"

epochs=50

model_cfgs=(
    "vitb32_openclip_laion400m" \
    # "vitb16_bioclip_treeoflife10m" \
    # "resnet50_imagenet_pretrained" \
    # "resnet50_inat_pretrained" \
    # "dinov2_vitb14_reg" \
    # "vitb16_openclip_laion400m" \
    # "vitb32_imagenet_pretrained" \
    # "resnet50_scratch" \
    #  "resnet50_clip" \
    # "dinov2_vits14_reg" \
    # "dinov2_vitb14_reg" \
    # "dinov2_vitl14_reg" \
    # "dinov2_vitg14_reg" \
    # "dinov3_vitb16" \
    # "vitb32_clip_inat" \
    # "vitb32_clip_nabirds" \
    # "vitb32_clip_cub" \
    )

log_mode="both"

#------------------------------
# DO NOT MODIFY BELOW THIS LINE !!!
#------------------------------

for model_cfg in "${model_cfgs[@]}"; do

    # update learning rate based on the first item of the model_cfg
    first_item=$(echo $model_cfg | cut -d'_' -f1)
    second_item=$(echo $model_cfg | cut -d'_' -f2)
    echo "Model Config: $model_cfg"
    # echo "First Item: $first_item"
    # echo "Second Item: $second_item"

    if [[ "$first_item" = "resnet50" && ( "$second_item" = "imagenet" || "$second_item" = "inat" ) ]]; then
        lr_classifier=1e-3
        wd=1e-2
        cls_inits=("random")
        optim="AdamW"
        temp_scheme_list=('none')
        temperature_list=(1.0)

    elif [ "$first_item" = "vitb32" ] && [ "$second_item" = "imagenet" ]; then
        lr_classifier=1e-3
        wd=1e-2
        cls_inits=("random")
        optim="AdamW"
        temp_scheme_list=('none')
        temperature_list=(1.0)

    elif [ "$first_item" = "resnet50" ] && [ "$second_item" = "clip" ]; then
        lr_classifier=1e-3
        wd=1e-2
        cls_inits=("REAL-Prompt")
        optim="AdamW"

    elif [[ "$first_item" = "vitb32" || "$first_item" = "vitb16" ]] && [[ "$second_item" = "openclip" || "$second_item" = "bioclip" ]]; then
        lr_classifier=1e-4
        wd=1e-2
        cls_inits=("REAL-Prompt")
        optim="AdamW"
        temp_scheme_list=('fewshot+retrieved+unlabeled') # has to tune temperature!!
        temperature_list=(0.07)
        # temp_scheme_list=('none') # this leads to poor performance of 43.9, which is barely improved above zero-shot 43.8
        # temperature_list=(1.0)

    elif [[ "$first_item" = "dinov2" || "$first_item" = "dinov3" ]]; then
        lr_classifier=1e-4
        wd=1e-2
        cls_inits=("random")
        optim="AdamW"
        temp_scheme_list=('none')
        temperature_list=(1.0)        

    else
        echo "Model not found"
        exit 1
    fi


    # update folder by adding the model_cfg
    folder="${prefix}_${model_cfg}_${epochs}epochs"
    echo "Folder: $folder"

    # Split the model_cfg by underscore and get the second item
    second_item=$(echo $model_cfg | cut -d'_' -f2)

    if [ "$second_item" = "openclip" ] || [ "$second_item" = "clip" ] || [ "$second_item" = "bioclip" ]; then
        script="main.py"
    else
        script="main_ssl.py"
    fi

    # Check if command-line arguments were provided
    if [ "$#" -ge 2 ]; then
        datasets=("$1")  # Use the provided command-line argument for the dataset
        seeds=("$2")
    else
        echo "Usage: $0 <dataset> [seed]"
    fi


    # Check if the results folder exists, if not create it
    if [ ! -d "results/$folder" ]; then
        mkdir -p "results/$folder"
    fi

    output_folder="output/$folder"
    if [ ! -d "$output_folder" ]; then
        mkdir -p "$output_folder"
    fi


    # Dynamically set the filename based on the dataset
    output_file="results/${folder}/${datasets[0]}_seed${seeds[0]}.csv"

    # Create or clear the output file
    echo "Dataset,Method,Model,DataSource,Cls_init,Shots,Seed,Retrieval_split,Temp_scheme,Temp,Stage1_Acc,Stage2_LPAcc,Stage2_FTAcc" > "$output_file"

    # Loop through all combinations and run the script
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            for data_source in "${data_sources[@]}"; do
                for shots in "${shot_values[@]}"; do
                    for init in "${cls_inits[@]}"; do
                        for seed in "${seeds[@]}"; do
                            for temp_scheme in "${temp_scheme_list[@]}"; do
                                for temperature in "${temperature_list[@]}"; do

                                    echo "Running: $script $model_cfg $method $dataset $data_source $init $shots $seed"

                                    # Run the script and capture the output
                                    output=$(python -W ignore "$script" --dataset "$dataset" --method "$method" \
                                            --data_source "$data_source"  --cls_init "$init" \
                                            --shots "$shots" --seed "$seed" --epochs "$epochs" --bsz "$batch_size" \
                                            --lr_classifier "$lr_classifier"  \
                                            --wd "$wd" --optim "$optim" --loss_name "$loss"\
                                            --log_mode "$log_mode" \
                                            --model_cfg "$model_cfg" --folder "$output_folder" \
                                            --temp_scheme "$temp_scheme" --temperature "$temperature" \
                                            --skip_stage2 \
                                            --recal_fea \
                                            --recal_prompt \
                                            --check_zeroshot \
                                            # --scale_text_embedding \
                                            )

                                    # Print the output to the console
                                    echo "$output"

                                    # Append the results to the CSV file
                                    echo "$output" >> "$output_file"

                                    echo ""
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done