import os
import shutil
import pandas as pd
import json


def get_image_count_per_class(split):
    df = pd.read_csv(f'classification_{split}.csv')
    print(f"Processing classification_{split}.csv with {len(df)} entries.")

    # loop through each row
    class_info = {}

    for idx, row in df.iterrows():
        species = row['standardized_species']
        family = row['family']
        if species not in class_info:
            class_info[species] = {}
            class_info[species]['name'] = species
            class_info[species]['family'] = family
            class_info[species]['count'] = 1
        else:
            class_info[species]['count'] += 1

    # sort the dictionary by count in increasing order
    class_info = dict(sorted(class_info.items(), key=lambda item: item[1]['count']))
    print(f"Found {len(class_info)} unique species in classification_{split}.csv.")

    return class_info


if __name__ == "__main__":

    for split in ['train', 'val', 'test']:
        count_dict = get_image_count_per_class(split)
        # save to a json file
        with open(f'image_count_{split}.json', 'w') as f:
            json.dump(count_dict, f, indent=4)
        print(f"Saved image count per class to image_count_{split}.json")
