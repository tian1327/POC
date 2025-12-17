QueryLMM for post-hoc correction
===============================


###  Qwen-2.5-VL-7B-Instruct

(1) Run Top-5 Inference Using Qwen-2.5-VL-7B-Instruct

You can choose dataset to be either of these: ```semi-aves```, ```fungitastic-m```, ```species196_insecta```, ```species196_weeds```, ```species196_mollusca```. You can pick the ```topk-json``` corresponding to the expert model (for e.g., Fewshot-FT, FineR, CLAP, CMLP, etc.) and backbone (e.g., OpenCLIP, DINOv2, iNat-pretrained ResNet-50, etc.) you want to use . We show the example for the 16-shot Fewshot-FT model and the OpenCLIP-ViT-B/32 backbone.  Don't forget to update ```dataset_path``` in ```config.yml``` with your dataset path (folder where you store the datasets).

You should update the ```prompt-template``` based on what prompting method you want to use. Check ```TEMPLATE_MAP``` in ```run_inference_local_hf.py``` for more information. Update the number of shots in the arguments for ```prompt-template``` and ```ref-image-dir``` accordingly.

```bash
python run_inference_local_hf.py \
  --prompt-template top5-multimodal-16shot-with-confidence_ranking \
  --prompt-dir semi-aves \
  --backend huggingface \
  --hf-model-name-or-path Qwen/Qwen2.5-VL-7B-Instruct \
  --config-yaml ../config.yml \
  --image-dir semi-aves \
  --image-paths ../data/semi-aves/test.txt \
  --ref-image-dir ../path/to/your/semi-aves/pregenerated_references_16shot \
  --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
  --topk-json ../path/to/your/fewshot_finetune_vitb32_openclip_laion400m_semi-aves_16_1_topk_test_predictions.json \
  --output-csv ../path/to/your/output.csv \
  --error-file ../path/to/your/error.txt \
  --max_new_tokens 900

```

(2) Example of zeroshot prompting LMM

```bash
python run_inference_local_hf.py \
  --prompt-template zeroshot \
  --prompt-dir semi-aves \
  --backend huggingface \
  --hf-model-name-or-path Qwen/Qwen2.5-VL-7B-Instruct \
  --config-yaml ../config.yml \
  --image-dir semi-aves \
  --image-paths ../data/semi-aves/test.txt \
  --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
  --output-csv ../path/to/your/output.csv \
  --error-file ../path/to/your/error.txt \
  --max_new_tokens 900
```

(3) Run Top-K Inference Using Qwen-2.5-VL-7B-Instruct

Update the ```prompt-template``` and ```topk``` arguments accordingly. K can be 3, 5, 7, 10, 15.

```bash
python run_inference_local_hf_topk.py \
  --prompt-template top15-multimodal-16shot-with-confidence_ranking \
  --topk 15 \
  --prompt-dir semi-aves \
  --backend huggingface \
  --hf-model-name-or-path Qwen/Qwen2.5-VL-7B-Instruct \
  --config-yaml ../config.yml \
  --image-dir semi-aves \
  --image-paths ../data/semi-aves/test4000.txt \
  --ref-image-dir ../path/to/your/semi-aves/pregenerated_references_16shot \
  --taxonomy-json ../data/semi-aves/semi-aves_labels.json \
  --topk-json .../path/to/your/finetune_vitb32_openclip_laion400m_semi-aves_16_1_topk_test_predictions.json \
  --output-csv ../path/to/your/output.csv \
  --error-file ../path/to/your/error.txt \
  --max_new_tokens 900
```

---

### GLM-4.1V-9B-Thinking

(1) Run Top-5 Inference Using GLM-4.1V-9B-Thinking

```bash
python glm_inference.py \
  --prompt-template top5-multimodal-16shot-with-confidence_ranking \
  --prompt-dir fungitastic-m \
  --backend huggingface \
  --hf-model-name-or-path THUDM/GLM-4.1V-9B-Thinking \
  --config-yaml ../config.yml \
  --image-dir fungitastic-m \
  --image-paths ../data/fungitastic-m/test.txt \
  --ref-image-dir ../path/to/your/fungitastic-m/pregenerated_references_16shot \
  --taxonomy-json ../data/fungitastic-m/fungitastic-m_labels.json \
  --topk-json ../path/to/your/fewshot_finetune_vitb32_openclip_laion400m_fungitastic-m_16_1_topk_test_predictions.json \
  --output-csv ../path/to/your/output.csv \
  --error-file ../path/to/your/error.txt \
  --max_new_tokens 1200

```

(2) Example of zeroshot prompting LMM

```bash
python glm_inference.py \
  --prompt-template zeroshot \
  --prompt-dir fungitastic-m \
  --backend huggingface \
  --hf-model-name-or-path THUDM/GLM-4.1V-9B-Thinking \
  --config-yaml ../config.yml \
  --image-dir fungitastic-m \
  --image-paths ../data/fungitastic-m/test.txt \
  --taxonomy-json ../data/fungitastic-m/fungitastic-m_labels.json \
  --output-csv ../path/to/your/output.csv \
  --error-file ../path/to/your/error.txt \
  --max_new_tokens 1200
```


---

### GPT-5-mini


(1) Make sure you create a .env file within the  ``` lmm-inference ```  folder and store your ```OPENAI_API_KEY```

```bash
python gpt_inference.py \
  --prompt-template top5-multimodal-16shot-with-confidence_ranking \
  --prompt-dir species196_mollusca \
  --backend huggingface \
  --hf-model-name-or-path gpt-5-mini \
  --config-yaml ../config.yml \
  --image-dir species196_mollusca \
  --image-paths ../data/species196_mollusca/test.txt \
  --ref-image-dir ../path/to/your/species196_mollusca/pregenerated_references_16shot \
  --taxonomy-json ../data/species196_mollusca/species196_mollusca_labels.json \
  --topk-json ../path/to/your/fewshot_finetune_vitb32_openclip_laion400m_species196_mollusca_16_1_topk_test_predictions.json \
  --output-csv ../path/to/your/output.csv \
  --error-file ../path/to/your/error.tsv \
  --throttle-sec 0.5
```

(2) Example of zeroshot prompting

```bash
python gpt_inference.py \
  --prompt-template zeroshot \
  --prompt-dir species196_mollusca \
  --backend huggingface \
  --hf-model-name-or-path gpt-5-mini \
  --config-yaml ../config.yml \
  --image-dir species196_mollusca \
  --image-paths ../data/species196_mollusca/test.txt \
  --taxonomy-json ../data/species196_mollusca/species196_mollusca_labels.json \
  --output-csv ../path/to/your/output.csv \
  --error-file ../path/to/your/error.tsv \
  --throttle-sec 0.5
```