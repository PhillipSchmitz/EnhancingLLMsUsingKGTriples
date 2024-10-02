import json
import csv

# Read the JSON data from the file
with open('../data/entities.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Define the CSV file path
csv_file = 'entities.csv'

# Write data to CSV (only if both label and description are not empty)
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['ID', 'label', 'description']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for item_id, item_data in data.items():
        label = item_data.get('label', '')
        description = item_data.get('description', '')
        if label.strip() and description.strip():
            writer.writerow({'ID': item_id, 'label': label, 'description': description})

print(f"CSV file '{csv_file}' created successfully!")

# Read the JSON data from the file
with open('../data/relations.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Define the CSV file path
csv_file = 'relations.csv'

# Write data to CSV (only if both label and description are not empty)
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['ID', 'label', 'description']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for item_id, item_data in data.items():
        label = item_data.get('label', '')
        description = item_data.get('description', '')
        if label.strip() and description.strip():
            writer.writerow({'ID': item_id, 'label': label, 'description': description})

print(f"CSV file '{csv_file}' created successfully!")

# Read the JSON data from the file
with open('../data/types.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Define the CSV file path
csv_file = 'types.csv'

# Write data to CSV (only if both label and description are not empty)
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['ID', 'label', 'description']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for item_id, item_data in data.items():
        label = item_data.get('label', '')
        description = item_data.get('description', '')
        if label.strip() and description.strip():
            writer.writerow({'ID': item_id, 'label': label, 'description': description})

print(f"CSV file '{csv_file}' created successfully!")

