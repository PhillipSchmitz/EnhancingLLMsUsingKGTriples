import json
import pandas as pd
from collections import Counter, defaultdict
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import squarify
import itertools
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_triples(file_path):
    triples = []
    with open(file_path, 'r') as file:
        for line in file:
            triples.append(tuple(line.strip().split('\t')))
    return triples


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_samples(file_path):
    samples = []
    with open(file_path, 'r') as file:
        for line in file:
            samples.append(line.strip().split('\t'))
    return samples


def load_and_count_samples(pos_file_path, neg_file_path):
    with open(pos_file_path, 'r') as pos_file, open(neg_file_path, 'r') as neg_file:
        pos_samples = pos_file.readlines()
        neg_samples = neg_file.readlines()
    return len(pos_samples), len(neg_samples)


def extract_features(samples):
    entities = [h for h, _, t in samples]
    relations = [r for _, r, _ in samples]
    return entities, relations


def count_frequencies(entities, relations):
    entity_freq = Counter(entities)
    relation_freq = Counter(relations)
    return entity_freq, relation_freq


def plot_frequencies(entity_freq, relation_freq, title_prefix):
    logging.info(f"Plotting frequencies for {title_prefix}...")

    # Plot entity frequencies
    fig = go.Figure()
    pos_keys = list(entity_freq[0].keys())
    neg_keys = list(entity_freq[1].keys())
    fig.add_trace(go.Bar(x=pos_keys, y=list(entity_freq[0].values()), name='Positive Entities', marker_color='blue'))
    fig.add_trace(go.Bar(x=neg_keys, y=list(entity_freq[1].values()), name='Negative Entities', marker_color='red'))
    fig.update_layout(title=f'{title_prefix}: Positive vs. Negative Entity Frequencies', barmode='group')
    fig.show()

    # Plot relation frequencies
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(relation_freq[0].keys()), y=list(relation_freq[0].values()), name='Positive Relations',
                         marker_color='blue'))
    fig.add_trace(go.Bar(x=list(relation_freq[1].keys()), y=list(relation_freq[1].values()), name='Negative Relations',
                         marker_color='red'))
    fig.update_layout(title=f'{title_prefix}: Positive vs. Negative Relation Frequencies', barmode='group')
    fig.show()


def plot_top_entities_and_relations(entities_list, relations_list):
    logging.info("Plotting top 10 entities and relations by frequency")

    # Count frequencies
    entity_freq = Counter(entities_list)
    relation_freq = Counter(relations_list)

    # Plot top 10 entities
    top_entities = entity_freq.most_common(10)
    plt.figure(figsize=(10, 5))
    plt.bar([e[0] for e in top_entities], [e[1] for e in top_entities])
    plt.title('Top 10 Entities by Frequency')
    plt.xticks(rotation=90)
    plt.show()

    # Plot top 10 relations
    top_relations = relation_freq.most_common(10)
    plt.figure(figsize=(10, 5))
    plt.bar([r[0] for r in top_relations], [r[1] for r in top_relations])
    plt.title('Top 10 Relations by Frequency')
    plt.xticks(rotation=90)
    plt.show()


def validate_data(triples, entities, relations, dataset_type):
    logging.info(f"Validating {dataset_type} dataset")
    invalid_entries = 0
    for h, r, t in triples:
        if h not in entities or t not in entities or r not in relations:
            invalid_entries += 1
    logging.info(f"There are {invalid_entries} invalid entries in the {dataset_type} dataset.")
    return invalid_entries


def check_missing_labels(triples, entities, relations):
    missing_labels_count = 0
    missing_labels = []
    for h, r, t in triples:
        if h not in entities or not entities[h].get('label'):
            missing_labels_count += 1
            missing_labels.append(h)
        if t not in entities or not entities[t].get('label'):
            missing_labels_count += 1
            missing_labels.append(t)
        if r not in relations or not relations[r].get('label'):
            missing_labels_count += 1
            missing_labels.append(r)
    logging.info(f"There are {missing_labels_count} Wikidata IDs with missing labels")
    logging.info(f"Information on the label is missing for the following entities/relations: {missing_labels}")
    return missing_labels_count, missing_labels


def analyze_entity_types(entity_types):
    num_types_per_entity = [len(types) for types in entity_types.values()]
    avg_types_per_entity = sum(num_types_per_entity) / len(num_types_per_entity)
    logging.info(f"Average number of types per entity: {avg_types_per_entity}")

    all_types = [t for types in entity_types.values() for t in types]
    type_distribution = Counter(all_types)
    logging.info(f"Top 10 most common types: {type_distribution.most_common(10)}")

    single_type_count = sum(1 for types in entity_types.values() if len(types) == 1)
    multi_type_count = sum(1 for types in entity_types.values() if len(types) > 1)
    logging.info(f"Number of entities with a single type: {single_type_count}")
    logging.info(f"Number of entities with multiple types: {multi_type_count}")


def plot_co_occurrence(entity_types):
    logging.info("Creating and plotting co-occurrence matrix for entity types...")
    co_occurrence = defaultdict(lambda: defaultdict(int))
    for types in entity_types.values():
        for t1, t2 in itertools.combinations(types, 2):
            co_occurrence[t1][t2] += 1
            co_occurrence[t2][t1] += 1

    # Flatten the co_occurrence defaultdict structure into a list of tuples
    data = [(type1, type2, count) for type1, related_types in co_occurrence.items() for type2, count in
            related_types.items()]

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Type1', 'Type2', 'CoOccurrenceCount'])
    #print(df)

    logging.info("Generate a Treemap visualization for the matrix...")
    # Assuming df is your DataFrame
    df_grouped = df.groupby('Type1')['CoOccurrenceCount'].sum().reset_index()
    df_grouped = df_grouped.sort_values(by='CoOccurrenceCount', ascending=False)

    # Select the top n Type1 strings
    top_n = 20  # Adjust if needed
    df_top_n = df_grouped.head(top_n).copy()

    # Modify the labels to include the cumulative sum
    df_top_n.loc[:, 'Label'] = df_top_n.apply(lambda row: f"{row['Type1']} ({row['CoOccurrenceCount']})", axis=1)

    # Generate unique colors
    colors = plt.cm.tab20.colors  # You can choose any colormap you like

    # Create the Treemap
    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=df_top_n['CoOccurrenceCount'], label=df_top_n['Label'], color=colors[:top_n], alpha=0.8)
    plt.axis('off')
    plt.title(f"Top {top_n} Entity Types - Co-Occurrence")
    plt.show()


def visualize_network(triples):
    logging.info("Visualizing network")
    G = nx.DiGraph()
    G.add_edges_from([(t[0], t[2], {'relation': t[1]}) for t in triples])
    plt.figure(figsize=(12, 12))
    nx.draw(G, with_labels=True, node_size=20, font_size=10)
    plt.show()


def main():
    logging.info("Starting main function")

    # Define paths
    train_triples_path = '../data/train.txt'
    pos_samples_val_path = '../data/valid.txt'
    neg_samples_val_path = '../data/valid_negatives.txt'
    pos_samples_test_path = '../data/test.txt'
    neg_samples_test_path = '../data/test_negatives.txt'

    entities_path = '../data/entities.json'
    relations_path = '../data/relations.json'
    entity_types_path = '../data/entity2types.json'

    logging.info("Loading and reading files...")
    # Load data
    train_triples = load_triples(train_triples_path)
    entities = load_json(entities_path)
    relations = load_json(relations_path)
    entity_types = load_json(entity_types_path)

    pos_samples_val = load_samples(pos_samples_val_path)
    neg_samples_val = load_samples(neg_samples_val_path)
    pos_samples_test = load_samples(pos_samples_test_path)
    neg_samples_test = load_samples(neg_samples_test_path)

    logging.info("Checking for missing labels in training dataset...")
    # Determine missing labels of Wikidata IDs in the training dataset (train.txt)
    check_missing_labels(train_triples, entities, relations)

    logging.info("Computing the number of unique entities/relations...")
    # Count unique entities and relations
    entities_list = [t[0] for t in train_triples] + [t[2] for t in train_triples]
    relations_list = [t[1] for t in train_triples]

    unique_entities = set(entities_list)
    unique_relations = set(relations_list)

    logging.info(f"There is a total of {len(unique_entities)} unique entities and {len(unique_relations)} unique relations")

    # Plot top 10 entities and relations
    plot_top_entities_and_relations(entities_list, relations_list)

    logging.info("Determining sample distribution in test and validation datasets...")
    # Print sample counts
    pos_count_val, neg_count_val = load_and_count_samples(pos_samples_val_path, neg_samples_val_path)
    pos_count_test, neg_count_test = load_and_count_samples(pos_samples_test_path, neg_samples_test_path)

    logging.info(f"Validation set: Positive samples = {pos_count_val}, Negative samples = {neg_count_val}")
    logging.info(f"Test set: Positive samples = {pos_count_test}, Negative samples = {neg_count_test}")

    logging.info("Calculating sample frequencies in test and validation datasets...")
    # Extract and count features
    pos_entities_val, pos_relations_val = extract_features(pos_samples_val)
    neg_entities_val, neg_relations_val = extract_features(neg_samples_val)
    pos_entities_test, pos_relations_test = extract_features(pos_samples_test)
    neg_entities_test, neg_relations_test = extract_features(neg_samples_test)

    pos_entity_freq_val, pos_relation_freq_val = count_frequencies(pos_entities_val, pos_relations_val)
    neg_entity_freq_val, neg_relation_freq_val = count_frequencies(neg_entities_val, neg_relations_val)
    pos_entity_freq_test, pos_relation_freq_test = count_frequencies(pos_entities_test, pos_relations_test)
    neg_entity_freq_test, neg_relation_freq_test = count_frequencies(neg_entities_test, neg_relations_test)

    # Plot frequencies
    plot_frequencies((pos_entity_freq_val, neg_entity_freq_val), (pos_relation_freq_val, neg_relation_freq_val),
                     'Validation')
    plot_frequencies((pos_entity_freq_test, neg_entity_freq_test), (pos_relation_freq_test, neg_relation_freq_test),
                     'Test')

    logging.info("Checking for missing labels test and validation datasets...")
    # Validate data
    invalid_entries_val = validate_data(load_triples(pos_samples_val_path) + load_triples(neg_samples_val_path),
                                        entities, relations, dataset_type='validation')
    invalid_entries_test = validate_data(load_triples(pos_samples_test_path) + load_triples(neg_samples_test_path),
                                         entities, relations, dataset_type='test')

    logging.info("Analyzing entity types...")
    # Analyze entity types
    analyze_entity_types(entity_types)

    # Filter and visualize
    specific_type = 'Q5'  # Example type
    entities_of_specific_type = [entity for entity, types in entity_types.items() if specific_type in types]
    logging.info(f"Number of entities of type {specific_type}: {len(entities_of_specific_type)}")

    specific_relation = 'P106'  # Example relation
    entities_with_specific_relation = [t for t in train_triples if t[1] == specific_relation]
    logging.info(f"Number of triples with relation {specific_relation}: {len(entities_with_specific_relation)}")

    # Plot co-occurrence matrix
    plot_co_occurrence(entity_types)

    # NetworkX visualization
    #visualize_network(train_triples)

    logging.info("Ending main function")


if __name__ == '__main__':
    main()