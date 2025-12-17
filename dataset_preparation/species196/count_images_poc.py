import os
import json

if __name__ == "__main__":
    for split in ['insecta', 'weeds', 'mollusca']:

        directory = f'../species196_{split}/train'

        count_dict = {}
        # loop through each folder in the directory
        for folder in os.listdir(directory):
            if folder not in count_dict:
                count_dict[folder] = {}

            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                num_images = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                count_dict[folder]['train'] = num_images
            
            # get the number of images in the corresponding val folder
            val_folder_path = folder_path.replace('train', 'val')
            if os.path.isdir(val_folder_path):
                num_images = len([f for f in os.listdir(val_folder_path) if os.path.isfile(os.path.join(val_folder_path, f))])
                count_dict[folder]['val'] = num_images
            else:
                count_dict[folder]['val'] = 0

            # calculate total
            count_dict[folder]['total'] = count_dict[folder]['train'] + count_dict[folder]['val']

            # remove the folder if total is less than 20
            if count_dict[folder]['total'] < 20:
                print(f"Removing {folder} with total images {count_dict[folder]['total']}")
                del count_dict[folder]

                # delete the folder from both train and val folders
                for subfolder in ['train', 'val']:
                    parent_dir = os.path.join(directory.replace('train', subfolder))
                    folder_path = os.path.join(parent_dir, folder)
                    if os.path.isdir(folder_path):
                        # remove the entire folder and its contents
                        os.system(f'rm -rf "{folder_path}"')
                        print(f"Removed folder: {folder_path}")
                    else:
                        print(f"Folder does not exist: {folder_path}")

        # sort the count_dict by train ascending
        count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]['train']))
        # count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]['total']))

        # write out to a json file
        with open(f'count_{split}_images.json', 'w') as f:
            json.dump(count_dict, f, indent=4)
        print(f"Saved count_{split}_images.json, {len(count_dict)} categories.")
