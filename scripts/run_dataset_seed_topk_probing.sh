#!/bin/bash

# Define arrays of values for each parameter

prefix="topk"

# methods=("mixup" "saliencymix" "CMO" "cutmix-fs" "resizemix" "CMLP" "probing" "finetune" "FLYP" "cutmix" "fixmatch")
methods=("probing")

# data_sources=("fewshot" "retrieved" "fewshot+retrieved" "fewshot+unlabeled" "fewshot+retrieved+unlabeled")
data_sources=("fewshot")

# shot_values=(16 8 4)
shot_values=(16)

batch_size=32

loss="CE"

epochs=20

model_cfgs=(
    "vitb32_openclip_laion400m" \
    # "vitb16_bioclip_treeoflife10m" \
    # "resnet50_imagenet_pretrained" \
    # "resnet50_inat_pretrained" \
    # "dinov2_vitb14_reg" \
    # "vitb32_imagenet_pretrained" \
    # "resnet50_scratch" \
    # "vitb16_openclip_laion400m" \
    # "resnet50_clip" \
    # "vitb32_clip_inat" \
    # "vitb32_clip_nabirds" \
    # "vitb32_clip_cub" \
    # "dinov2_vits14_reg" \
    # "dinov2_vitl14_reg" \
    # "dinov2_vitg14_reg" \
    # "dinov3_vitb16" \
    )

log_mode="both"


#------------------------------
# DO NOT MODIFY BELOW THIS LINE !!!
#------------------------------

for model_cfg in "${model_cfgs[@]}"; do

    # update learning rate based on the first item of the model_cfg
    first_item=$(echo $model_cfg | cut -d'_' -f1)
    second_item=$(echo $model_cfg | cut -d'_' -f2)
    echo "Model: $model_cfg"
    # echo "First item: $first_item"
    # echo "Second item: $second_item"

    # resnet50_imagenet_pretrained
    if [ "$first_item" = "resnet50" ] && [ "$second_item" = "imagenet" ]; then
        lr_classifier=1e-3
        lr_backbone=1e-3
        wd=1e-4
        cls_inits=("random")
        optim="SGD"
        temp_scheme_list=('none')
        temperature_list=(1.0)        

    # resnet50_inat_pretrained
    elif [ "$first_item" = "resnet50" ] && [ "$second_item" = "inat" ]; then
        lr_classifier=1e-3
        lr_backbone=1e-3
        wd=1e-4
        cls_inits=("random")
        optim="SGD"
        temp_scheme_list=('none')
        temperature_list=(1.0)          

    # vitb32_imagenet_pretrained
    elif [ "$first_item" = "vitb32" ] && [ "$second_item" = "imagenet" ]; then
        lr_classifier=1e-3
        lr_backbone=1e-3
        wd=1e-4
        cls_inits=("random")
        optim="SGD"
        temp_scheme_list=('none')
        temperature_list=(1.0)          

    # resnet50_clip
    elif [ "$first_item" = "resnet50" ] && [ "$second_item" = "clip" ]; then
        lr_classifier=1e-4
        lr_backbone=1e-6
        wd=1e-2
        cls_inits=("REAL-Prompt")
        optim="AdamW"
        temp_scheme_list=('fewshot+retrieved+unlabeled') # has to tune temperature!!
        temperature_list=(0.07)        

    # openclip or clip
    elif [[ "$first_item" = "vitb32" || "$first_item" = "vitb16" ]] && [[ "$second_item" = "openclip" || "$second_item" = "bioclip" ]]; then
        lr_classifier=1e-4
        lr_backbone=1e-6
        wd=1e-2
        cls_inits=("REAL-Prompt")
        optim="AdamW"
        temp_scheme_list=('fewshot+retrieved+unlabeled') # has to tune temperature!!
        temperature_list=(0.07)

    # DINOv2 or DINOv3
    elif [[ "$first_item" = "dinov2" || "$first_item" = "dinov3" ]]; then
        lr_classifier=1e-4
        lr_backbone=1e-6
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
    folder="${prefix}_${methods[@]}_${model_cfg}_${epochs}epochs"
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

                                    echo "Running: $script $dataset $method $loss $model_cfg $data_source $init $shots $seed $retrieval_split $unlabeled_in_split"

                                    # set the model_path based on method
                                            #  "output/LinearProbing_vitb32_openclip_laion400m_50epochs/output_semi-aves/semi-aves_probing_fewshot_REAL-Prompt_16shots_seed1/stage1_model_best.pth"
                                    model_path="output/LinearProbing_${model_cfg}_50epochs/output_${dataset}/${dataset}_${method}_fewshot_${init}_${shots}shots_seed${seed}/stage1_model_best.pth"

                                    # Run the script and capture the output
                                    output=$(python -W ignore "$script" --prefix "$prefix" --dataset "$dataset" --method "$method" \
                                            --data_source "$data_source" --cls_init "$init" \
                                            --shots "$shots" --seed "$seed" --epochs "$epochs" --bsz "$batch_size" \
                                            --lr_classifier "$lr_classifier"  --lr_backbone "$lr_backbone" --wd "$wd" \
                                            --optim "$optim" --loss_name "$loss"\
                                            --log_mode "$log_mode" \
                                            --temp_scheme "$temp_scheme" --temperature "$temperature" \
                                            --model_cfg "$model_cfg" --folder "$output_folder" \
                                            --skip_stage2 \
                                            --model_path "$model_path" \
                                            --topk_predictions \
                                            # --cls_path "$cls_path" \
                                            # --check_zeroshot \
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