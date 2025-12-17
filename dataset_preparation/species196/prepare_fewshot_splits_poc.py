import os
import json
import random

if __name__ == "__main__":

    # prepare the train.txt and val.txt files based on the images in the folders
    for split in ['insecta', 'weeds', 'mollusca']:
        
        labels = json.load(open(f'../../post-hoc_correction/data/species196_{split}/species196_{split}_labels.json'))
        
        train_list = []
        val_list = []
        train_images_ct = 0
        val_images_ct = 0
        for id, info in labels.items():
            folder = info['folder_name']

            train_directory = f'../species196_{split}/train/' # train/ folder
            train_images = os.listdir(os.path.join(train_directory, folder))
            num_images = len(train_images)
            train_images_ct += num_images
            for img in train_images:
                img_path = os.path.join('train', folder, img)
                train_list.append(f"{img_path} {id} 1")

            val_directory = f'../species196_{split}/val/' # val/ folder
            val_images = os.listdir(os.path.join(val_directory, folder))
            num_images = len(val_images)
            val_images_ct += num_images
            for img in val_images:
                img_path = os.path.join('val', folder, img)
                val_list.append(f"{img_path} {id} 1")

        # save the train_list and val_list to a .txt file
        with open(f'../../post-hoc_correction/data/species196_{split}/train.txt', 'w') as f:
            f.write("\n".join(train_list))
        with open(f'../../post-hoc_correction/data/species196_{split}/val.txt', 'w') as f:
            f.write("\n".join(val_list))

        print(f'species196_{split}: train images: {train_images_ct}, val images: {val_images_ct}')

        
        #---------- sample the few-shot splits

        # collect line by class
        train = dict()
        val = dict()

        # for weeds and mollusca, since the train split has > 16 images per class, we sample few-shot from train split only
        # for insecta, since some classes have < 16 images in train split, we first augment train split with val split (exclusively), 
        # then sample few-shot from the augmented train split

        for line in train_list:
            path, class_id, source = line.strip('\n').split(' ')
            if class_id in train:
                train[class_id].append(path)
            else:
                train[class_id] = [path]

        # get val list
        for line in val_list:
            path, class_id, source = line.strip('\n').split(' ')
            if class_id in val:
                val[class_id].append(path)
            else:
                val[class_id] = [path]        

        # for class that has <16 train images, sample (16-train_ct) from val images
        for class_id, paths in train.items():
            if len(paths) < 16:
                needed = 16 - len(paths)
                if len(val[class_id]) < needed:
                    raise ValueError(f"Error: class {class_id} has only {len(val[class_id])} samples in val, less than needed {needed} samples.")
                else:
                    sampled = random.sample(val[class_id], needed)
                    train[class_id] += sampled # add to train set
                    val[class_id] = [p for p in val[class_id] if p not in sampled] # remove from val set

        # save the updated val set as the test set
        # get val_list again
        val_list = []
        for class_id, paths in val.items():
            for path in paths:
                val_list.append(f"{path} {class_id} 1")

        with open(f'../../post-hoc_correction/data/species196_{split}/test.txt', 'w') as f:
            f.write("\n".join(val_list))            

        # now we have enough images in train set to sample few-shot
        for seed in [1, 2, 3]:
            for shots in [4, 8, 16]:
                random.seed(seed)
                fewshot = dict()
                for class_id, paths in train.items():
                    if len(paths) < shots:
                        raise ValueError(f"Error: class {class_id} has only {len(paths)} samples, less than {shots} shots.")
                    else:
                        fewshot[class_id] = random.sample(paths, shots)
                
                # save to a txt file
                fewshot_list = []
                for class_id, paths in fewshot.items():
                    for path in paths:
                        fewshot_list.append(f"{path} {class_id} 1")

                with open(f'../../post-hoc_correction/data/species196_{split}/fewshot{shots}_seed{seed}.txt', 'w') as f:
                    f.write("\n".join(fewshot_list))