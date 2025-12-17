<div align="center">
<h1>Surely Large Multimodal Models (<em>Don't</em>) Excel<br>in Visual Species Recognition?</h1>

[**Tian Liu**](https://tian1327.github.io/)<sup>1</sup> · [**Anwesha Basu**](https://www.linkedin.com/in/anweshabasu98/)<sup>1</sup> · [**James Caverlee**](https://people.engr.tamu.edu/caverlee/index.html)<sup>1</sup> · [**Shu Kong**](https://aimerykong.github.io/)<sup>2</sup>

<sup>1</sup>Texas A&M University&emsp;&emsp;&emsp;<sup>2</sup>University of Macau
<br>
<!-- &dagger;project lead&emsp;*corresponding author -->

<a href="https://arxiv.org/abs/xxxx"><img src='https://img.shields.io/badge/arXiv-POC-red' alt='Paper PDF'></a>
<a href='https://tian1327.github.io/POC/'><img src='https://img.shields.io/badge/Project_Page-POC-green' alt='Project Page'></a>
</div>

<!-- Our work adapts a pretrained Vision-Language Model (VLM) and retrieves relevant pretraining images to solve few-shot recognition problem.
To mitigate the `domain gap` and `imbalanced distribution` problems of retrieved data, we propose a novel **Stage-Wise retrieval-Augmented fineTuning (SWAT)** method, which outperforms previous few-shot recognition methods by >6% in accuracy across nine benchmark datasets. -->

<div align="center">

![teaser](assets/POC_teaser.png)

</div>

## News

- **2025-12-16:** POC code released.

<!-- - **2025-12-06:** We release pre-created `laion400m.db` file for easy retrieval. See [RETRIEVAL.md](./retrieval/RETRIEVAL.md).
- **2025-05-27:** SWAT is accepted to 4th CVinW and FGVC12 workshops at CVPR'25! 
- **2025-02-26:** SWAT is accepted to CVPR 2025! ;)
- **2025-01-18:** We provide access to our retrieved data through URLs. See [RETRIEVAL.md](./retrieval/RETRIEVAL.md).
- **2024-11-24:** Updated code base to include more datasets.
- **2024-08-22:** Retrieval code released, see [RETRIEVAL.md](./retrieval/RETRIEVAL.md).
- **2024-07-05:** SWAT finetuning code released.
- **2024-06-28:** [project page](https://tian1327.github.io/SWAT/) launched.
- **2024-06-17:** [arXiv paper](https://arxiv.org/abs/2406.11148) released. -->

## Create Environment

Create conda environment and install dependencies.

```bash
# lab server has CUDA version 12.8, thus using pytorch-cuda=12.1 for compatibility
# DINOv3 requries python=3.10

conda create -n poc python=3.10 -y
conda activate poc
conda install pytorch torchvision torchaudio torchmetrics pytorch-cuda=12.1 -c pytorch -c nvidia

# install openclip and clip
pip install open_clip_torch
pip install git+https://github.com/openai/CLIP.git

pip install pandas scikit-learn 

# clone dinov3
git clone https://github.com/facebookresearch/dinov3.git

# install gdown for downloading datasets
pip install gdown
```

For LMM inference with Qwen, you can follow [instructions](https://github.com/QwenLM/Qwen2.5-VL) or steps below to set up Qwen2.5-VL-7B locally.

```bash
# setup Qwen2.5-VL-7B locally using huggingface transformers
conda create --name qwen --clone poc
conda activate qwen
pip install transformers==4.51.3 accelerate
pip install qwen-vl-utils[decord]
```


## Dataset Prepraration

Prepare the datasets following the instructions in [DATASETS.md](./DATASETS.md).


## Code Usage

1. Obtain the top-k predictions on the test set using a few-shot finetuned models.

```bash
# activate conda environment
. env_s2.sh
conda activate poc

# few-shot linear probing
bash scripts/run_dataset_seed_probing.sh semi-aves 1

# few-shot finetuning
bash scripts/run_dataset_seed_fewshot_finetune.sh semi-aves 1

# obtain top-k predictions on test set for a pretrained model
bash scripts/run_dataset_seed_topk.sh semi-aves 1

# we can also run batch experiments for multiple datasets and seeds
bash scripts/batch_probing.sh
bash scripts/batch_fewshot_finetune.sh
bash scripts/batch_topk.sh

# run other FSL baselines, we will release more FSL baselines soon
bash scripts/batch_cmlp.sh

```

For running ```FineR```, follow the command given below. Note this is just FineR and not POC on top of FineR. You can change the number of shots to 4, 8 or 16. Update the ```train_list``` and ```output_json``` arguments accordingly.

```bash

# Activate your environment
conda activate poc

# Obtain FineR predictions. 
python finer_topk.py \
  --model_cfg ViT-B-32 \
  --pretrained laion400m_e32 \
  --device cuda \
  --dataset_name semi-aves \
  --dataset_root_train ../path/to/your/semi-aves/ \
  --dataset_root_test  ../path/to/your/semi-aves/ \
  --train_list data/semi-aves/fewshot4_seed1.txt \
  --test_list  data/semi-aves/test.txt \
  --metrics_json data/semi-aves/semi-aves_labels.json \
  --name_key most_common_name \
  --use_random_aug --aug_repeats 10 --encode_chunk_size 512 \
  --alpha 0.7 \
  --logit_scale_eval 1.0 \
  --logit_scale_export 50 \
  --batch_size 256 --workers 8 \
  --topk 10 \
  --output_json ../path/to/your/fineR_semi-aves_4shot_topk_fused.json
```

2. Query LMM for post-hoc correction.

```bash
conda activate qwen

cd post-hoc_correction/lmm-inference

# run the query script
```

See [QUERYLMM.md](./QUERYLMM.md). for instructions on running query with each LMM. 


## Citation

If you find our project useful, please consider citing our related works:

```bibtex



@inproceedings{liu2025few,
    title={Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning},
    author={Liu, Tian and Zhang, Huixin and Parashar, Shubham and Kong, Shu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}

@inproceedings{parashar2024neglected,
    title={The Neglected Tails in Vision-Language Models},
    author={Parashar, Shubham and Lin, Zhiqiu and Liu, Tian and Dong, Xiangjue and Li, Yanan and Ramanan, Deva and Caverlee, James and Kong, Shu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}

```
