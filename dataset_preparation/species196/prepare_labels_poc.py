import pandas as pd
import os
import json

if __name__ == "__main__":

    taxonomy = pd.read_csv("Taxonomy.csv")
    # create a new column 'folder' whose value is the first word of the 'Scientific Name' column
    taxonomy['folder'] = taxonomy['Scientific name'].apply(lambda x: x.split(' ')[0])
    # print(taxonomy.head())
    # print(taxonomy.info())
    
    # fill the NaN values in the 'Common name' column with the value in the 'Scientific Name' column
    taxonomy['Common name'] = taxonomy['Common name'].fillna(taxonomy['Scientific name'])
    # print(taxonomy.info())

    for split in ['insecta', 'weeds', 'mollusca']:

        labels_dict = {}
        directory = f'../species196_{split}/train/'
        folders = sorted(os.listdir(directory))
        for i, folder in enumerate(folders):
            if folder not in taxonomy['folder'].values:
                raise ValueError(f"Folder {folder} not found in taxonomy.csv")
            else:
                scientific_name = taxonomy[taxonomy['folder'] == folder]['Scientific name'].values[0]
                most_common_name = taxonomy[taxonomy['folder'] == folder]['Common name'].values[0]
                class_name = taxonomy[taxonomy['folder'] == folder]['Class'].values[0]
                order = taxonomy[taxonomy['folder'] == folder]['Order'].values[0]
                family = taxonomy[taxonomy['folder'] == folder]['Family'].values[0]
                genus = taxonomy[taxonomy['folder'] == folder]['Genus'].values[0]


                labels_dict[str(i)] = {
                    'name': scientific_name,
                    'folder_name': folder,
                    'most_common_name': most_common_name,
                    'alternates': {
                        scientific_name: 0,
                        most_common_name: 0
                    },
                    'class': class_name,
                    'order': order,
                    'family': family,
                    'genus': genus
                    }
                
        # save the labels_dict to a json file
        path = f'../../post-hoc_correction/data/species196_{split}/species196_{split}_labels.json'
        with open(path, 'w') as f:
            json.dump(labels_dict, f, indent=4)
        print(f"Saved {path}")





