import json
import random

if __name__ == "__main__":
    
    # merge the count_train+val_images_only_testset.json and count_test_images.json

    with open(f'count_test_images.json', 'r') as f:
        species_info = json.load(f)
    
    with open(f'count_train+val_images_only_testset.json', 'r') as f:
        train_val_species_info = json.load(f)
    
    for species in train_val_species_info:
        if species not in species_info:
            raise ValueError(f"Species {species} in train+val not found in test set")
        else:
            species_info[species]['filenames'] = train_val_species_info[species]['filenames'] + species_info[species]['filenames']
            species_info[species]['count'] = len(species_info[species]['filenames'])

    # exclude the corrupted images from the filenames
    with open('corrupted_images.txt', 'r') as f:
        corrupted_images = f.read().splitlines()
    # converted to set for faster lookup
    corrupted_images = set(corrupted_images)

    print(f"Excluding {len(corrupted_images)} corrupted images from the new split")
    for species in species_info:
        original_count = species_info[species]['count']
        species_info[species]['filenames'] = [f for f in species_info[species]['filenames'] if f not in corrupted_images]
        species_info[species]['count'] = len(species_info[species]['filenames'])
        if species_info[species]['count'] < original_count:
            print(f"Species {species}: {original_count} -> {species_info[species]['count']} after excluding corrupted images")
        if species_info[species]['count'] < 4:
            raise ValueError(f"Species {species} has less than 4 images after excluding corrupted images, cannot sample for test set")

    # sort the species_info by count
    species_info = dict(sorted(species_info.items(), key=lambda item: item[1]['count']))
    # write out to a json file
    with open(f'count_train+val+test_images.json', 'w') as f:
        json.dump(species_info, f, indent=4)
    print(f"Saved count_train+val+test_images.json, {len(species_info)} categories.")

    #---------- prepare the labels.json file
    labels_dict = {}
    for i, species in enumerate(species_info):
        labels_dict[str(i)] = {
            'name': species,
            'most_common_name': species,
            # 'count': species_info[species]['count'],
            # 'filenames': species_info[species]['filenames'],
            "class": species_info[species]['class'],
            'order': species_info[species]['order'],
            'family': species_info[species]['family'],
            'genus': species_info[species]['genus'],
            'alternates': {species: 0}
        }

    with open(f'../../post-hoc_correction/data/fungitastic-m/fungitastic-m_labels.json', 'w') as f:   
        json.dump(labels_dict, f, indent=4)
    print(f"Saved fungitastic-m_labels.json, {len(labels_dict)} categories.")

    #---------- prepare the fewshot splits files
    # sample at least 4 images, at most 20 images per species for testing
    test_split = []
    train_split = []
    # set random seed for reproducibility
    random.seed(42)
    for i, species in enumerate(species_info):
        filenames = species_info[species]['filenames']
        if len(filenames) < 4:
            raise ValueError(f"Species {species} has less than 4 images, cannot sample for test set")
        elif len(filenames) <= 20:
            species_info[species]['test_filenames'] = random.sample(filenames, 4)
            species_info[species]['train_filenames'] = [f for f in filenames if f not in species_info[species]['test_filenames']]
        elif len(filenames) <= 36:
            test_filenames = random.sample(filenames, len(filenames)-16)
            species_info[species]['test_filenames'] = test_filenames
            species_info[species]['train_filenames'] = [f for f in filenames if f not in test_filenames]
        else:
            test_filenames = random.sample(filenames, 20)
            species_info[species]['test_filenames'] = test_filenames
            species_info[species]['train_filenames'] = [f for f in filenames if f not in test_filenames]
        
        for file in species_info[species]['test_filenames']:
            test_split.append(f"{file} {i} 1")
        for file in species_info[species]['train_filenames']:
            train_split.append(f"{file} {i} 1")
    
    # write out the updated species_info to a json file
    with open(f'count_train+val+test_images_newsplit.json', 'w') as f:
        json.dump(species_info, f, indent=4)
    print(f"Saved count_train+val+test_images_newsplit.json, {len(species_info)} categories.")

    # write out the train and test splits to text files
    with open(f'../../post-hoc_correction/data/fungitastic-m/train.txt', 'w') as f:
        f.write("\n".join(train_split))
    print(f"Saved train.txt, {len(train_split)} images.")
    with open(f'../../post-hoc_correction/data/fungitastic-m/test.txt', 'w') as f:
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
            with open(f'../../post-hoc_correction/data/fungitastic-m/fewshot{kshot}_seed{seed}.txt', 'w') as f:
                f.write("\n".join(fewshot_split))
            print(f"Saved fewshot{kshot}_seed{seed}.txt, {len(fewshot_split)} images.")

