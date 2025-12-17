import json
import os

if __name__ == "__main__":
    for split in ['insecta', 'weeds', 'mollusca']:
        
        print(f"\nRemoving duplicate categories for {split} ...")

        if split == 'insecta':
            duplicate_categories = ['Cydia', 'Rhynchophorus', 'Phenacoccus', 'Rhabdoscelus', 'Rhynchites', 
                                    'Scolytus', 'Carpomya', 'Dysmicoccus', 'Eurytoma', 'Hoplocampa', 
                                    'Lepidosaphes', 'Anthonomus', '\u6b27\u6d32Rhynchites', '\u65e5\u672cRhynchites']
        elif split == 'weeds':
            duplicate_categories = ['Solanum', 'Avena', 'Aegilops', 'Centaurea', 'Emex', 
                                    'Eupatorium', 'Iva', 'Lactuca', 'Sorghum']
        elif split == 'mollusca':
            duplicate_categories = ['Helix']
        

        directory = f'../species196_{split}'
        # removed the folder from both train and val folders
        for subfolder in ['train', 'val']:
            print(f"Processing {subfolder} folder...")
            parent_dir = os.path.join(directory, subfolder)
            for folder in duplicate_categories:
                folder_path = os.path.join(parent_dir, folder)
                if os.path.isdir(folder_path):
                    # remove the entire folder and its contents
                    os.system(f'rm -rf "{folder_path}"')
                    print(f"Removed folder: {folder_path}")
                else:
                    print(f"Folder does not exist: {folder_path}")
        
