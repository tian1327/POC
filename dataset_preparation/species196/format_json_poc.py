import json

# load the species196L_train.json file
with open('species196L_train.json', 'r') as f:
    data = json.load(f)

# dump it out to a new file with indentation for better readability
with open('species196L_train_pretty.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Formatted JSON saved to species196L_train_pretty.json")

# do the same thing for species196L_val.json
with open('species196L_val.json', 'r') as f:
    data = json.load(f)
with open('species196L_val_pretty.json', 'w') as f:
    json.dump(data, f, indent=4)    
print("Formatted JSON saved to species196L_val_pretty.json")