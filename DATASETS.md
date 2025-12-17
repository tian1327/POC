# How to install datasets

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like:

```
$DATA/
|–– semi-aves/
|–– species196_insecta/
|–– species196_weeds/
|–– species196_mollusca/
|–– fungitastic-m/
|–– fish-vista-m/
```

Update the `config.yml` with the path to the datasets.

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:

- [Semi-Aves](#semi-aves)
- [Species196](#species196)
- [FungiTastic](#fungitastic)
- [Fish-Vista](#fish-vista)

The instructions to prepare each dataset are detailed below. The corresponding scripts are available in the [dataset_preparation](#dataset_preparation) folder.

### Semi-Aves

- Create a folder named `semi-aves/` under `$DATA`.
- Download data from the [official repository](https://github.com/cvl-umass/semi-inat-2020) or following the `gdown` commands below

```bash
cd $DATA/semi-aves/

# train_val data
gdown https://drive.google.com/uc?id=1xsgOcEWKG9CszNNT_EXN3YB1OLPYNbf8 

# test
gdown https://drive.google.com/uc?id=1OVEA2lNJnYM5zxh3W_o_Q6K5lsNmJ9Hy

# unlabeled_ID
gdown https://drive.google.com/uc?id=1BiEkIp8yuqB5Vau_ZAFAhwp8CNoqVjRk

# unzip
tar -xzf *.gz
```

- The annotations are extracted from the [official annotation json files](https://github.com/cvl-umass/semi-inat-2020). We have reformatted and provided to you as `ltrain.txt`, `ltrain+val.txt`,`val.txt` and `test.txt` in the `data/semi-aves/` folder.
- Note that the official test set contains 8000 images (40 images per class for 200 classes). We randomly sampled 20 images from each class to form a smaller test set of 4000 images for faster evaluation during training. The `test.txt` file provided in the `data/semi-aves/` folder corresponds to this smaller test set. The full test set is also available in the `test_8000.txt`.
  
The directory structure should look like:

```
semi-aves/
|–– trainval_images
|–– u_train_in
|–– test
```

### Species196

- download meta data from [website](https://species-dataset.github.io/download.html)
- clean up invalid downloaded images
- separate the insecta, weeds, mollusca into three sub datasets, i.e., insecta, weeds, mollusca
- exclude the ambiguoius classes
- choose only classes with at least 20 images after combining train and val set
- generate the labels
- generate the few-shot and test split for running experiments

```bash
cd Species196_L_v0.1/

# format the annotation files for better readability
python format_json_poc.py

# download images using urls
python download_imgnet_style_norepeat_poc.py

# clean up downloads by removing non-image files, and fix suffix to .jpg
python cleanup_downloads_poc.py

# copy images to species196-insecta, species196-weeds, species196-mollusca
python organize_species196_poc.py 

# note there are some duplicated/ambiguious categories in the annotation files,
# I simply exclude those ambiguous categories
python check_annotation_poc.py

python remove_duplicate_classes_poc.py

# for species196_insecta, remove the classes that have less than total 20 images in train+val set
python count_images_poc.py

# generate label file
python prepare_labels_poc.py

# prepare the few-shot and test split files
python prepare_fewshot_splits_poc.py

```


### FungiTastic

- follow instructions on [website](https://bohemianvra.github.io/FungiTastic/#downloading-the-data) to download the data and metadata (including labels) 
 - the data is also available on the [webpage](https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download/)

```bash
# note that I updated the download.py to ensure successful downloading
git clone https://github.com/poc1327/FungiTastic.git
cd FungiTastic/dataset

# download the FungiTastic-Mini dataset
# note that the fullsize images contain many image files that are truncated,
# which can lead to errors when loading the data in dataloaders.
# Instead, the smaller-resolution versions have less this issue. 
# Hence, we download the 720p version.
mkdir fungitastic-m
python download.py --metadata --images --subset "m" --size "720" --save_path "./fungitastic-m"

# note the metadata downloaded does not contain labels for the test set,
# hence we need to manually download the `metadata_test_labels.zip` from: 
# https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download/

# move folder around to organize the files
cd /FungiTastic/dataset/fungitastic-m/FungiTastic/FungiTastic-Mini
mv train ../../
mv val ../../
mv test ../../
mv dna-test ../../

cd ..
mv metadata ../

cd ..
mv fungitastic-m/ ../../dataset/

# count the number of images per species for each split
cd dataset/fungitastic-m
python count_images_poc.py

# get a list of corrupted images to exclude in the training
# in the train/720p folder, there are 750 corrupted images out of the total 46842 images
python check_corrupted_images_poc.py

# 1. Combine train, val, and test sets for creating new train and test splits
# this is to ensure sufficient number of images for train and test.
# 2. Exclude the corrupted images from the new split.
# 3. Each class has at least 4 images, at most 20 images for testing.
# The fewshot images are then sampled from the train split.
# The label.json file is also generated.
python prepare_fewshot_labels_poc.py

```


### Fish-Vista

> This dataset is not explored in our experiments, but we provide instructions for downloading and preparing the data for future use.

- Follow download instruction from [huggingface](https://huggingface.co/datasets/imageomics/fish-vista). Note, `git lfs` installation is needed, which requires sudo access.

```bash
# many images are downloaded from git lfs automatically when cloning the huggingface repo
git clone https://huggingface.co/datasets/imageomics/fish-vista
cd fish-vista/

mkdir AllImages
find Images -type f -exec mv -v {} AllImages \;
rm -rf Images
mv AllImages Images

# create env
conda create -n fish-vista python=3.10 
conda activate fish-vista
pip install -r requirements.txt

# download the copyrighted images
python download_and_process_nd_images.py --save_dir Images

# check the availability of downloaded images and move them to train/val/test folders
python check_downloaded_images_poc.py

# compress the images and tranfer to remote server
tar -czvf fish-vista.tar.gz train val test
tar -xzvf fish-vista.tar.gz

# the file size is about 7.2GB
rsync -avz --progress fish-vista.tar.gz server_path

# prepare the labels and few-shot splits
python prepare_fewshot_labels_poc.py

```


<!-- ### FishNet

- image data can be downloaded from [webpage](https://fishnet-2023.github.io/)
- annotations can be downloaded from [repo](https://github.com/faixan-khan/FishNet/tree/main/anns)

```bash
# download images
cd $DATA/fishnet
gdown 1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3

# exclude species that have less than 20 images, 
# sample the bottom 100 species as the rare classes

``` -->

Transfer the prepared datasets to target server.

```bash
cd

# use rsync to transfer the datasets to target server
rsync -avz --progress dataset/ server_path

# use scp to transfer the datasets to local Mac
tar -czvf poc_datasets.tar.gz dataset/
scp file_path_on_server ~/Downloads/
tar -xzvf poc_datasets.tar.gz

```
