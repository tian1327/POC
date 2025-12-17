import os
import shutil
import pandas as pd
import json
import random


def get_image_count_per_class(split, class_info):
    df = pd.read_csv(f'classification_{split}.csv')
    print(f"Processing classification_{split}.csv with {len(df)} entries.")

    # loop through each row
    for idx, row in df.iterrows():
        species = row['standardized_species']
        family = row['family']
        filename = row['filename']
        new_filename = filename.replace(" ", "_")  # replace spaces with underscores
        # change the filename to use the new filename
        old_file = f"{split}/{filename}"
        new_file = f"{split}/{new_filename}"
        if os.path.exists(old_file):
            os.rename(old_file, new_file)

        filename = f"{split}/{new_filename}"
        if species not in class_info:
            class_info[species] = {}
            class_info[species]['name'] = species
            class_info[species]['family'] = family
            class_info[species]['total_count'] = 1
            class_info[species]['filenames'] = [filename]
        else:
            class_info[species]['total_count'] += 1
            class_info[species]['filenames'].append(filename)


if __name__ == "__main__":

    count_dict = {}
    for split in ['train', 'val', 'test']:
        get_image_count_per_class(split, count_dict)

    # sort the dictionary by count in increasing order
    count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]['total_count']))
    print(f"Found {len(count_dict)} unique species in classification_split.csv.")

    # save to a json file
    with open(f'image_count_train+val+test.json', 'w') as f:
        json.dump(count_dict, f, indent=4)
    print(f"Saved image count per class to image_count_train+val+test.json")

    # exclude species with less than 20 images
    filtered_count_dict = {k: v for k, v in count_dict.items() if v['total_count'] >= 20 and v['total_count'] <= 100}
    print(f"Filtered to {len(filtered_count_dict)} unique species with at least 20 images and at most 100 images.")

    # add the class, genus information using the "train.csv" file from FishNet dataset
    # https://github.com/faixan-khan/FishNet/anns
    train_df = pd.read_csv('train.csv')
    # convert the species names to lowercase
    train_df['species'] = train_df['species'].str.lower()

    no_matching_count = 0
    keys = list(filtered_count_dict.keys())
    for species in keys:
        matching_rows = train_df[train_df['species'] == species]
        if matching_rows.empty:
            # raise ValueError(f"No matching rows found for species: {species}")
            print(f"No matching rows found for species: {species}")
            no_matching_count += 1
            # remove the species from the dict
            del filtered_count_dict[species]
        else:
            class_name = matching_rows['Class'].values[0]
            order = matching_rows['Order'].values[0]
            family = matching_rows['Family'].values[0]
            genus = matching_rows['Genus'].values[0]
            filtered_count_dict[species]['class'] = class_name
            filtered_count_dict[species]['order'] = order
            filtered_count_dict[species]['family'] = family
            filtered_count_dict[species]['genus'] = genus

    print(f"Total species with no matching rows: {no_matching_count}")
    print(f"The length of dict is {len(filtered_count_dict)} after removing species with no matching rows.")

    with open(f'image_count_train+val+test_filtered.json', 'w') as f:
        json.dump(filtered_count_dict, f, indent=4)
    print(f"Saved filtered image count per class to image_count_train+val+test_filtered.json")

    # prepare the label file
    # use 0, 1, 2, 3 as the keys for the dict
    label_dict = {}
    for i, (k, v) in enumerate(filtered_count_dict.items()):
        label_dict[i] = {}
        label_dict[i]['name'] = v['name']
        label_dict[i]['most_common_name'] = v['name']
        label_dict[i]['class'] = v['class']
        label_dict[i]['order'] = v['order']
        label_dict[i]['family'] = v['family']
        label_dict[i]['genus'] = v['genus']
        label_dict[i]['alternates'] = {
            v['name']: 0
        }
    
    with open(f'fish-vista-m_labels.json', 'w') as f:
        json.dump(label_dict, f, indent=4)
    print(f"Saved labels to fish-vista-m_labels.json")
    # I did some manual cleaning on the label file to add the missing class and order names, and remove the '/misc'


    #---------- prepare the fewshot splits files
    # sample at least 4 images, at most 20 images per species for testing
    test_split = []
    train_split = []
    # set random seed for reproducibility
    random.seed(42)
    species_info = filtered_count_dict
    for i, species in enumerate(species_info):
        filenames = species_info[species]['filenames']
        if len(filenames) < 4:
            raise ValueError(f"Species {species} has less than 4 images, cannot sample for test set")
        elif len(filenames) <= 20:
            species_info[species]['test_filenames'] = random.sample(filenames, 4) # all species have at least 4 images
            species_info[species]['train_filenames'] = [f for f in filenames if f not in species_info[species]['test_filenames']]
        elif len(filenames) <= 36:
            test_filenames = random.sample(filenames, len(filenames)-16)
            species_info[species]['test_filenames'] = test_filenames
            species_info[species]['train_filenames'] = [f for f in filenames if f not in test_filenames]
        else:
            test_filenames = random.sample(filenames, 20) # max 20 images for test set
            species_info[species]['test_filenames'] = test_filenames
            species_info[species]['train_filenames'] = [f for f in filenames if f not in test_filenames]
        
        for file in species_info[species]['test_filenames']:
            test_split.append(f"{file} {i} 1")
        for file in species_info[species]['train_filenames']:
            train_split.append(f"{file} {i} 1")
    
    # write out the updated species_info to a json file
    with open(f'image_count_train+val+test_filtered_newsplit.json', 'w') as f:
        json.dump(species_info, f, indent=4)
    print(f"Saved image_count_train+val+test_filtered_newsplit.json, {len(species_info)} categories.")

    # write out the train and test splits to text files
    # with open(f'../../post-hoc_correction/data/fungitastic-m/train.txt', 'w') as f:
    with open(f'train.txt', 'w') as f:
        f.write("\n".join(train_split))
    print(f"Saved train.txt, {len(train_split)} images.")

    # with open(f'../../post-hoc_correction/data/fungitastic-m/test.txt', 'w') as f:
    with open(f'test.txt', 'w') as f:
        f.write("\n".join(test_split))
    print(f"Saved test.txt, {len(test_split)} images.")

    # sample 4/8/16 fewshot splits from the train set using three random seeds
    fewshot_seeds = [1, 2, 3]
    fewshot_kshots = [4, 8, 16]
    for seed in fewshot_seeds:
        random.seed(seed)
        for kshot in fewshot_kshots:
            fewshot_split = []
            for i, species in enumerate(species_info):
                train_filenames = species_info[species]['train_filenames']
                if len(train_filenames) < kshot:
                    sampled_filenames = train_filenames
                else:
                    sampled_filenames = random.sample(train_filenames, kshot)
                for file in sampled_filenames:
                    fewshot_split.append(f"{file} {i} 1")
            # write out the fewshot split to text file
            # with open(f'../../post-hoc_correction/data/fungitastic-m/fewshot{kshot}_seed{seed}.txt', 'w') as f:
            with open(f'fewshot{kshot}_seed{seed}.txt', 'w') as f:
                f.write("\n".join(fewshot_split))
            print(f"Saved fewshot{kshot}_seed{seed}.txt, {len(fewshot_split)} images.")

    