import json
import csv
import os

# Load the JSON data for entities and relations
with open('../data/entities.json', 'r', encoding='utf-8') as entities_file:
    entities_data = json.load(entities_file)

with open('../data/relations.json', 'r', encoding='utf-8') as relations_file:
    relations_data = json.load(relations_file)


# Function to get the label from the JSON data
def get_label(wikidata_id):
    if wikidata_id.startswith('Q'):
        return entities_data.get(wikidata_id, {}).get('label', '')
    elif wikidata_id.startswith('P'):
        return relations_data.get(wikidata_id, {}).get('label', '')
    return ''


# Input directory
text_files_directory = '../data'  # Adjust this path as needed

# List of text files to process
text_files = ['test.txt', 'test_negatives.txt', 'train.txt', 'valid.txt', 'valid_negatives.txt']

# Process each text file
for text_file in text_files:
    triples = []
    text_file_path = os.path.join(text_files_directory, text_file)
    with open(text_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            subject, predicate, obj = line.strip().split('\t')
            triples.append((subject, predicate, obj))

    # Create a corresponding CSV file in the current directory
    csv_file = os.path.splitext(text_file)[0] + '.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Subject', 'Predicate', 'Object'])  # Write header
        for subject, predicate, obj in triples:
            subject_label = get_label(subject)
            predicate_label = get_label(predicate)
            object_label = get_label(obj)
            csvwriter.writerow([subject_label, predicate_label, object_label])

    print(f"CSV file {csv_file} has been created successfully.")