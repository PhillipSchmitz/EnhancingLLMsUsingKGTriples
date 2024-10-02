from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
import torch
import pandas as pd

def load_data(file_path, num_samples=None):
    # Use the entire dataset
    df = pd.read_csv(file_path)
    # Optionally, limit the dataset to the specified number of samples
    if num_samples is not None:
        df = df[:num_samples]

    return df['Entity Label'].tolist(), df['Entity Description'].tolist()


def compute_start_end_positions(context, tokenizer):
    """Compute start and end positions of the answer in the context based on the custom rule."""
    try:
        # Step 1: Find the position of the first occurrence of the word ' ist '
        ist_index = context.find(' ist ')
        if ist_index == -1:
            raise ValueError("'ist' not found in context.")

        # Step 2: Calculate the start and end positions of the answer in character indices
        start_char = ist_index + len(' ist ')  # Start after 'ist '
        end_char = context.rfind('.')
        if end_char == -1:
            raise ValueError("No period found in context.")

        if start_char >= end_char:
            raise ValueError("Start position is greater than or equal to end position.")

        # Step 3: Encode the context with offsets to map characters to tokens
        context_encodings = tokenizer(context, return_offsets_mapping=True, truncation=True)

        # Debugging: Print the full encoding output
        #print(f"Context Encodings: {context_encodings}")

        # Extracting input IDs correctly
        input_ids = context_encodings['input_ids']

        # Debugging: Print token information for all tokens in the input
        for idx, token in enumerate(input_ids):
            token_str = tokenizer.convert_ids_to_tokens(token)
            start_offset, end_offset = context_encodings['offset_mapping'][idx]
            #print(f"Token: {token_str} | Position: {idx} | Start: {start_offset} | End: {end_offset}")

        start_position = None
        end_position = None

        # Map character positions to token positions
        for idx, (start, end) in enumerate(context_encodings['offset_mapping']):
            if start_position is None and start >= start_char:
                start_position = idx
            if end_position is None and end > end_char:
                end_position = idx - 1  # Adjust if the end token is not a complete word or is punctuation
                break

        if start_position is None or end_position is None:
            raise ValueError("Could not find valid start or end positions.")

        # Ensure end_position points to a valid token (i.e., not punctuation)
        while end_position > 0 and tokenizer.convert_ids_to_tokens(input_ids[end_position]) in [".", ",", ")", ""]:
            end_position -= 1

        # Debugging: Print calculated positions and related information
        #print(f"Context: {context}")
        #print(f"Start Character Position: {start_char}, End Character Position: {end_char}")
        #print(f"Calculated Start Token Position: {start_position}, Calculated End Token Position: {end_position}")

        return start_position, end_position

    except ValueError as e:
        print(f"Error computing positions: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, start_positions, end_positions):
        self.encodings = encodings
        self.start_positions = start_positions
        self.end_positions = end_positions

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['start_positions'] = torch.tensor(self.start_positions[idx])
        item['end_positions'] = torch.tensor(self.end_positions[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def train_model(dataset, tokenizer, output_dir='./results'):
    # Use the German BERT model
    model = BertForQuestionAnswering.from_pretrained('bert-base-german-cased')

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save model and tokenizer in the same directory
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

def evaluate_model(question, context, model, tokenizer, device):
    model.eval()

    # Tokenize the input question and context
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Identify the start and end positions of the answer
    answer_start = torch.argmax(outputs.start_logits).item()
    answer_end = torch.argmax(outputs.end_logits).item()

    # Ensure end position is not before start position
    if answer_end < answer_start:
        answer_end = answer_start

    # Use the tokenizer's offset mapping to get the exact spans
    offset_mapping = inputs['offset_mapping'][0]  # Get offset mapping for the first (and only) example
    start_char_position = offset_mapping[answer_start][0]  # Start character position
    end_char_position = offset_mapping[answer_end][1]      # End character position

    # Extract the answer from the context based on character positions
    answer = context[start_char_position:end_char_position].strip()

    return answer

# Main script
if __name__ == "__main__":
    # Load the data
    entity_labels, entity_descriptions = load_data('entity2types_2.0_sentences.csv', num_samples=None) # Specify number of samples, if necessary
    print("Ja")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')

    # Prepare start and end positions
    start_positions = []
    end_positions = []

    for description in entity_descriptions:
        start_pos, end_pos = compute_start_end_positions(description, tokenizer)
        if start_pos is None or end_pos is None:
            print(f"Skipping context due to invalid positions: {description}")
            continue
        start_positions.append(start_pos)
        end_positions.append(end_pos)

    # Tokenize the dataset
    encodings = tokenizer(entity_labels, entity_descriptions, truncation=True, padding=True, return_tensors='pt')

    dataset = QADataset(encodings, start_positions, end_positions)
    train_model(dataset, tokenizer)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the fine-tuned model and tokenizer for evaluation
    model = BertForQuestionAnswering.from_pretrained('./fine_tuned_model')
    tokenizer = BertTokenizerFast.from_pretrained('./fine_tuned_model')

    model = model.to(device)

    # Example evaluation
    question = "Was ist Filmschauspieler?"
    context = "Filmschauspieler (ID: Q10800557) ist Schauspieler, der in Filmen auftritt."
    answer = evaluate_model(question, context, model, tokenizer, device)

    print(f"Question: {question}")
    print(f"Answer: {answer}")