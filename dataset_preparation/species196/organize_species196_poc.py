import json
import os

if __name__ == "__main__":
    for split in ['insecta', 'weeds', 'mollusca']:
        
        print(f"\nProcessing {split} ...")
        
        # collect the category names from species196L_train_pretty.json
        with open('species196L_train_pretty.json', 'r') as f:
            data = json.load(f)
        category_names_train = [entry['name'] for entry in data['categories'] if entry['supercategory'].lower() == split] 
        category_names_train = sorted(category_names_train)
        # collect the category names from species196L_val_pretty.json
        with open('species196L_val_pretty.json', 'r') as f:
            data = json.load(f)
        category_names_val = [entry['name'] for entry in data['categories'] if entry['supercategory'].lower() == split]
        category_names_val = sorted(category_names_val)

        # ensure train and val are matching
        assert category_names_train == category_names_val, f"Train and val category names do not match for {split}"
        print(f"{split} category names match between train and val with {len(category_names_train)} categories.")

        # make a new directry for the split
        output_dir = f'../species196_{split}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # make the train and val directories
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        print(f"Created directories: {train_dir}, {val_dir}")

        # copy the folder for each category from the original train and val directories to the new directories
        original_train_dir = '../species196/train'
        original_val_dir = '../species196/val'
        for category in category_names_train:
            # copy from original train dir to new train dir
            src = os.path.join(original_train_dir, category)
            dst = os.path.join(train_dir, category)

            # copy the directory
            if os.path.exists(src):
                if not os.path.exists(dst):
                    os.system(f'cp -r "{src}" "{dst}"')
                    print(f"Copied {src} to {dst}")
                else:
                    print(f"Directory {dst} already exists, skipping copy.")
            else:
                print(f"Source directory {src} does not exist, skipping.")
        
            # copy from original val dir to new val dir
            src = os.path.join(original_val_dir, category)
            dst = os.path.join(val_dir, category)

            # copy the directory
            if os.path.exists(src):
                if not os.path.exists(dst):
                    os.system(f'cp -r "{src}" "{dst}"')
                    print(f"Copied {src} to {dst}")
                else:
                    print(f"Directory {dst} already exists, skipping copy.")
            else:
                print(f"Source directory {src} does not exist, skipping.")

print("\nAll splits processed.")