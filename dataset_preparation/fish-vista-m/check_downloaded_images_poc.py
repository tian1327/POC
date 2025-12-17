import os
import shutil
import pandas as pd


if __name__ == "__main__":

    for split in ['train', 'val', 'test']:
        df = pd.read_csv(f'classification_{split}.csv')
        print(f"Processing classification_{split}.csv with {len(df)} entries.")

        missing_images = []
        for idx, row in df.iterrows():
            image_path = 'Images/' + row['filename']
            if not os.path.exists(image_path):
                missing_images.append(image_path)

        if missing_images:
            print(f"Missing {len(missing_images)} images in classification_{split}.csv:")
            # write missing images to a text file
            with open(f'missing_images_{split}.txt', 'w') as f:
                for img in missing_images:
                    f.write(f"{img}\n")
        else:
            print(f"All images are present for classification_{split}.csv.")
    
        # move the files to a folder names split
        if not os.path.exists(f'{split}'):
            os.makedirs(f'{split}')

        for idx, row in df.iterrows():
            image_path = 'Images/' + row['filename']
            if os.path.exists(image_path):
                shutil.move(image_path, f'{split}/{row['filename']}')
        
        # print the number of files in the split folder
        num_files = len(os.listdir(f'{split}'))
        print(f"Moved {num_files} files to the {split} folder.")