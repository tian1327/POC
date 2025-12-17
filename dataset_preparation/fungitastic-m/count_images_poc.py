import os
import json
import pandas as pd
import copy

if __name__ == "__main__":

    # combine the train and val set to ensure sufficient number of images per species
    file_list = [
        'FungiTastic-Mini-Train.csv',
        'FungiTastic-Mini-Val.csv',
    ]

    folder = ['train/720p/', 'val/720p/']
    species_info = dict()

    for file, folder in zip(file_list, folder):
        path = f'metadata/FungiTastic-Mini/{file}'
        df = pd.read_csv(path)
        
        # loop through each row in the df
        for index, row in df.iterrows():
            scientific_name = row['scientificName']
            class_name = row['class']
            order = row['order']
            family = row['family']
            genus = row['genus']
            filename = folder+row['filename']
                        
            if scientific_name not in species_info:
                species_info[scientific_name] = {
                    'name': scientific_name,
                    'class': class_name,
                    'genus': genus,
                    'family': family,
                    'order': order,
                    'count': 1,
                    'filenames': [filename],
                }
            else:
                species_info[scientific_name]['count'] += 1
                species_info[scientific_name]['filenames'].append(filename)
            
    # sort the species_info by count
    species_info = dict(sorted(species_info.items(), key=lambda item: item[1]['count']))
    # write out to a json file
    with open(f'count_train+val_images.json', 'w') as f:
        json.dump(species_info, f, indent=4)
    print(f"Saved count_train+val_images.json, {len(species_info)} categories.")

    train_val_species_info = copy.deepcopy(species_info)

    # test set
    file_list = [
        'FungiTastic-Mini-Test.csv',
    ]
    folder = ['test/720p/']
    species_info = dict()

    for file, folder in zip(file_list, folder):
        path = f'metadata/FungiTastic-Mini/{file}'
        df = pd.read_csv(path)
        
        # loop through each row in the df
        for index, row in df.iterrows():
            scientific_name = row['scientificName']
            class_name = row['class']
            order = row['order']
            family = row['family']
            genus = row['genus']
            filename = folder+row['filename']
                        
            if scientific_name not in species_info:
                species_info[scientific_name] = {
                    'name': scientific_name,       
                    'class': class_name,             
                    'genus': genus,
                    'family': family,
                    'order': order,
                    'count': 1,
                    'filenames': [filename],
                }
            else:
                species_info[scientific_name]['count'] += 1
                species_info[scientific_name]['filenames'].append(filename)
            
    # sort the species_info by count
    species_info = dict(sorted(species_info.items(), key=lambda item: item[1]['count']))
    # write out to a json file
    with open(f'count_test_images.json', 'w') as f:
        json.dump(species_info, f, indent=4)
    print(f"Saved count_test_images.json, {len(species_info)} categories.")


    # remove the species in the train_val_species_info that are not in the test set
    species_to_remove = []
    for species in train_val_species_info:
        if species not in species_info:
            species_to_remove.append(species)
    print(f"Removing {len(species_to_remove)} species that are not in the test set: {species_to_remove}")
    for species in species_to_remove:
        del train_val_species_info[species]
    
    with open(f'count_train+val_images_only_testset.json', 'w') as f:
        json.dump(train_val_species_info, f, indent=4)
    print(f"Saved count_train+val_images_only_testset.json, {len(train_val_species_info)} categories.")
