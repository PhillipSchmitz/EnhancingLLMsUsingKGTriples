import pandas as pd

def create_sentences(df):
    """Create unique entity and related sentences based on the DataFrame."""
    entity_data = []  # This will store tuples of (Entity Label, Entity Description)
    unique_sentences = set()  # Set to track unique sentences

    for index, row in df.iterrows():
        entity_id = row['Entity ID']
        entity_label = row['Entity Label']
        entity_description = row['Entity Description']
        related_id = row['Related ID']
        related_label = row['Related Label']
        related_description = row['Related Description']

        # Create the entity's own description sentence
        entity_sentence = f"{entity_label} ist {entity_description}."

        # Add sentence with the entity as the subject if it's unique
        if entity_sentence not in unique_sentences:
            entity_data.append((entity_label, entity_sentence))
            unique_sentences.add(entity_sentence)

        # Create the related description sentence
        related_sentence = (
            f"Ein verwandter Begriff von {entity_label} ist {related_label}, "
            f"der als {related_description} beschrieben werden kann."
        )

        # Use Related Label as Entity Label for related sentences and ensure uniqueness
        if related_sentence not in unique_sentences:
            entity_data.append((related_label, related_sentence))
            unique_sentences.add(related_sentence)

    return entity_data

def write_to_csv(entity_data, output_file):
    """Write the entity-label/description pairs to a new CSV file."""
    descriptions_df = pd.DataFrame(entity_data, columns=['Entity Label', 'Entity Description'])
    descriptions_df.to_csv(output_file, index=False, encoding='utf-8')

# Main script
if __name__ == "__main__":
    # Define file paths
    input_csv_file = 'entity2types_2.0.csv'
    new_csv_file = 'entity2types_2.0_sentences.csv'

    # Execute the workflow
    df = pd.read_csv(input_csv_file)
    entity_data = create_sentences(df)
    write_to_csv(entity_data, new_csv_file)

    print(f"CSV file '{new_csv_file}' created successfully!")