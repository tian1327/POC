import json

if __name__ == "__main__":

    # load the species196L_train_pretty.json file
    for prefix in ['train', 'val']:
        with open(f'species196L_{prefix}_pretty.json', 'r') as f:
            data = json.load(f)

        # check the categories field for any duplications
        insecta_category_count = {}
        weeds_category_count = {}
        mollusca_category_count = {}

        for entry in data['categories']:
            if entry['supercategory'] == 'Insecta':
                if entry['name'] in insecta_category_count:
                    insecta_category_count[entry['name']] += 1
                else:
                    insecta_category_count[entry['name']] = 1
            elif entry['supercategory'] == 'Weeds':
                if entry['name'] in weeds_category_count:
                    weeds_category_count[entry['name']] += 1
                else:
                    weeds_category_count[entry['name']] = 1
            elif entry['supercategory'] == 'Mollusca':
                if entry['name'] in mollusca_category_count:
                    mollusca_category_count[entry['name']] += 1
                else:
                    mollusca_category_count[entry['name']] = 1

        # sort the category count by count descending
        insecta_category_count = dict(sorted(insecta_category_count.items(), key=lambda item: item[1], reverse=True))
        weeds_category_count = dict(sorted(weeds_category_count.items(), key=lambda item: item[1], reverse=True))
        mollusca_category_count = dict(sorted(mollusca_category_count.items(), key=lambda item: item[1], reverse=True))

        # write out insecta_category_count to a json file
        with open(f'{prefix}_insecta_category_count.json', 'w') as f:
            json.dump(insecta_category_count, f, indent=4)

        # write out weeds_category_count to a json file
        with open(f'{prefix}_weeds_category_count.json', 'w') as f:
            json.dump(weeds_category_count, f, indent=4)

        # write out mollusca_category_count to a json file
        with open(f'{prefix}_mollusca_category_count.json', 'w') as f:
            json.dump(mollusca_category_count, f, indent=4)
