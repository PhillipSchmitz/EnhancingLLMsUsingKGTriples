import json
import csv
import time
from qwikidata.linked_data_interface import get_entity_dict_from_api
import os

# Load JSON files
with open('../data/entities.json', 'r', encoding='utf-8') as f:
    entities = json.load(f)

with open('../data/relations.json', 'r', encoding='utf-8') as f:
    relations = json.load(f)

# Get the label for a given Wikidata ID
def get_label(wID, fetch_attempts):
    if wID.startswith('Q'):
        data = entities
    elif wID.startswith('P'):
        data = relations
    else:
        print(f"Error: Invalid Wikidata ID: {wID}")
        return None

    label = data.get(wID, {}).get('label', '')
    if not label:
        fetch_attempts += 1
        retries = 3
        for _ in range(retries):
            try:
                entity_dict = get_entity_dict_from_api(wID)
                label = entity_dict.get('labels', {}).get('de', {}).get('value') or entity_dict.get('labels', {}).get('en', {}).get('value')
                break
            except Exception as e:
                print(f"Error retrieving entity {wID}: {e}")
                time.sleep(5)
    return label, fetch_attempts

if __name__ == '__main__':
    text_files_directory = '../data'
    text_files = ['test.txt', 'test_negatives.txt', 'train.txt', 'valid.txt', 'valid_negatives.txt']

    for text_file in text_files:
        text_file_path = os.path.join(text_files_directory, text_file)
        fetch_attempts = 0

        with open(text_file_path, 'r', encoding='utf-8') as f:
            triples = f.readlines()

        csv_file = os.path.splitext(text_file)[0] + '_2.0.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Subject', 'Predicate', 'Object'])  # Write header
            for triple in triples:
                wIDs = triple.strip().split()
                labels = []
                for wID in wIDs:
                    label, fetch_attempts = get_label(wID, fetch_attempts)
                    labels.append(label)
                csvwriter.writerow(labels)

        print(f"The CSV file '{csv_file}' has been created.")
        print(f"There were {fetch_attempts} attempts to fetch data from Wikidata due to missing data in the input file.")