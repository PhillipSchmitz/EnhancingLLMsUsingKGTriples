import csv
import json

# Read the CSV files
entities_file = '../data_preprocessing/entities.csv'
types_file = '../data_preprocessing/types.csv'

# Initialize dictionaries to store entity information
entity_info = {}
related_info = {}

# Read entities CSV
with open(entities_file, 'r', encoding='utf-8') as entities_csv:
    reader = csv.reader(entities_csv)
    next(reader)  # Skip header row
    for row in reader:
        entity_id, label, description = row
        entity_info[entity_id] = {'label': label, 'description': description}

# Read types CSV
with open(types_file, 'r', encoding='utf-8') as types_csv:
    reader = csv.reader(types_csv)
    next(reader)  # Skip header row
    for row in reader:
        type_id, type_label, type_description = row
        related_info[type_id] = {'label': type_label, 'description': type_description}

# Read the JSON file
with open('../data/entity2types.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Initialize a list to store rows for the CSV
csv_rows = []

# Process each example
for entity_id, related_ids in data.items():
    entity_label = entity_info.get(entity_id, {}).get('label', 'Not found')
    entity_description = entity_info.get(entity_id, {}).get('description', 'Not found')
    for related_id in related_ids:
        related_label = related_info.get(related_id, {}).get('label', 'Not found')
        related_description = related_info.get(related_id, {}).get('description', 'Not found')
        csv_rows.append([entity_id, entity_label, entity_description, related_id, related_label, related_description])

# Write to a CSV file
output_file = 'entity2types.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Entity ID', 'Entity Label', 'Entity Description', 'Related ID', 'Related Label', 'Related Description'])
    writer.writerows(csv_rows)

print(f"CSV file '{output_file}' created successfully!")