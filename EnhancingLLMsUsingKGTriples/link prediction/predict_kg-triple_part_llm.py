from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class PartLabelMapper:
    def __init__(self):
        self.part_to_id = {}  # Map to store string part to ID
        self.next_id = 0

    def get_label_id(self, part_value):
        if part_value not in self.part_to_id:
            self.part_to_id[part_value] = self.next_id
            self.next_id += 1
        return self.part_to_id[part_value]

    def reverse_map_label_id(self, label_id):
        for part_value, id in self.part_to_id.items():
            if id == label_id:
                return part_value
        return "Unknown"


def read_triples(file_path, nrows=None):
    with open(file_path, "r", encoding="utf-8") as file:
        if nrows:
            return [next(file).strip() for _ in range(nrows)]
        else:
            return [line.strip() for line in file]


def process_triples(triples, mapper, triple_part_to_train_on):
    labels = []
    for triple in triples:
        subject, predicate, obj = triple.split(maxsplit=2)

        # Select which part to use for training based on the string variable
        if triple_part_to_train_on == "subject":
            label_id = mapper.get_label_id(subject)
        elif triple_part_to_train_on == "object":
            label_id = mapper.get_label_id(obj)
        else:  # Default to "predicate"
            label_id = mapper.get_label_id(predicate)

        labels.append(label_id)
    return labels


def tokenize_triples(triples, tokenizer, max_length):
    input_ids = []
    attention_mask = []
    for triple in triples:
        encoded = tokenizer.encode_plus(triple, add_special_tokens=True, max_length=max_length, padding="max_length",
                                        truncation=True)
        input_ids.append(encoded["input_ids"])
        attention_mask.append(encoded["attention_mask"])
    return torch.tensor(input_ids), torch.tensor(attention_mask)


def create_dataloader(input_ids, attention_mask, labels, batch_size):
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size)


def train_model(model, dataloader_train, dataloader_valid, optimizer, loss_fn, epochs, save_model_path=None, metrics_file_path=None):
    # Open file for saving metrics if a path is provided
    if metrics_file_path:
        metrics_file = open(metrics_file_path, "w")
        metrics_file.write("Epoch,Avg_Train_Loss,Validation_Loss,Validation_Accuracy\n")

    for epoch in range(epochs):
        model.train()
        total_batches = len(dataloader_train)
        running_loss = 0.0

        # Training loop
        for step, batch in enumerate(dataloader_train,1):
            b_input_ids, b_attention_mask, b_labels = batch
            optimizer.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, b_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 10 == 0 or step == total_batches:
                avg_loss = running_loss / step
                print(f"Epoch {epoch + 1}/{epochs} | Batch {step}/{total_batches} | Avg Loss: {avg_loss:.4f}")

        avg_loss = running_loss / len(dataloader_train)
        print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # Validation after each epoch
        validation_loss, validation_accuracy = validate_model(model, dataloader_valid, loss_fn)

        # Write metrics to file
        if metrics_file_path:
            metrics_file.write(f"{epoch + 1},{avg_loss:.4f},{validation_loss:.4f},{validation_accuracy:.4f}\n")

    # Close metrics file
    if metrics_file_path:
        metrics_file.close()

    # Save the trained model
    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

def load_model(model, load_model_path):
    model.load_state_dict(torch.load(load_model_path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model was loaded from {load_model_path}")
    return model

def validate_model(model, dataloader_valid, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader_valid:
            b_input_ids, b_attention_mask, b_labels = batch

            outputs = model(b_input_ids, attention_mask=b_attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader_valid)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Return validation loss and accuracy for logging
    return avg_loss, accuracy


def predict_triples(model, dataloader_test, mapper, tokenizer, triple_part_to_train_on):
    results = []
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for batch in dataloader_test:
            b_input_ids, b_attention_mask, b_labels = batch

            # Forward pass
            outputs = model(b_input_ids, attention_mask=b_attention_mask)
            logits = outputs.logits

            # Get predicted labels
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())

            # Reverse map predicted labels to selected part (subject, predicate, or object)
            for input_id, pred in zip(b_input_ids, preds):
                predicted_value = mapper.reverse_map_label_id(pred.item())
                input_triple = tokenizer.decode(input_id, skip_special_tokens=True)
                results.append((input_triple, predicted_value))

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate and plot confusion matrix (conditionally)
    if triple_part_to_train_on == "predicate":
        # Generate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Plot confusion matrix using Seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=mapper.predicate_to_id.keys(),
                    yticklabels=mapper.predicate_to_id.keys())
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for {triple_part_to_train_on.capitalize()}')
        plt.show()
    else:
        print(f"Confusion matrix skipped for {triple_part_to_train_on} due to large value range.")
        # Alternative: Calculate Top-k Accuracy and check whether the true label is in the top-k predictions
        top_k_accuracy = calculate_top_k_accuracy(logits, b_labels, k=5)
        print(f"Top-5 Accuracy: {top_k_accuracy:.4f}")

    return results

def calculate_top_k_accuracy(logits, labels, k=10): # Adapt k (Idea: Start with 5 or 10, and if needed set it to 20 or 50)
    top_k_preds = torch.topk(logits, k=k, dim=-1).indices
    correct = 0
    total = labels.size(0)

    for i in range(total):
        if labels[i] in top_k_preds[i]:
            correct += 1

    return correct / total

def write_results(results, batch_size, triple_part_to_train_on):
    with open(f"results_{triple_part_to_train_on}_bs{batch_size}.txt", "w", encoding="utf-8") as result_file:
        for triple, predicted_label in results:
            result_file.write(f"Triple: {triple}, Predicted Label: {predicted_label}\n")


# Main script
if __name__ == "__main__":
    # Define the part of the triple to train on (e.g., 'subject', 'predicate', 'object')
    triple_part_to_train_on = "predicate"  # Change this to the desired triple part

    mapper = PartLabelMapper()

    # Read the training, validation, and test data (positive samples only)
    train_triples = read_triples("../data/train.txt")
    valid_triples = read_triples("../data/valid.txt")
    test_triples = read_triples("../data/test.txt")

    # Process triples to extract the selected part (subject, predicate, or object)
    train_labels = process_triples(train_triples, mapper, triple_part_to_train_on)
    valid_labels = process_triples(valid_triples, mapper, triple_part_to_train_on)
    test_labels = process_triples(test_triples, mapper, triple_part_to_train_on)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=mapper.next_id)

    max_length = 64
    batch_size = 64

    # Tokenize the data
    train_input_ids, train_attention_mask = tokenize_triples(train_triples, tokenizer, max_length)
    valid_input_ids, valid_attention_mask = tokenize_triples(valid_triples, tokenizer, max_length)
    test_input_ids, test_attention_mask = tokenize_triples(test_triples, tokenizer, max_length)

    # Create DataLoader for training, validation, and testing data
    dataloader_train = create_dataloader(train_input_ids, train_attention_mask, train_labels, batch_size)
    dataloader_valid = create_dataloader(valid_input_ids, valid_attention_mask, valid_labels, batch_size)
    dataloader_test = create_dataloader(test_input_ids, test_attention_mask, test_labels, batch_size)

    print(f"Training labels: {set(train_labels)}")
    print(f"Validation labels: {set(valid_labels)}")
    print(f"Test labels: {set(test_labels)}")
    print(f"Number of unique labels: {mapper.next_id}")

    # Train the model
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define where to save the model and metrics (optional)
    save_model_path = f"bert_model_{triple_part_to_train_on}.pth"
    metrics_file_path = f"training_metrics_{triple_part_to_train_on}.txt"

    # Train the model (optional: save it, together with metrics)
    train_model(model, dataloader_train, dataloader_valid, optimizer, loss_fn, epochs=3,
                save_model_path=save_model_path, metrics_file_path=metrics_file_path)

    # Load the saved model for testing or prediction (optional)
    #model = load_model(model, save_model_path)

    # Evaluate on the test set
    results = predict_triples(model, dataloader_test, mapper, tokenizer, triple_part_to_train_on)

    # Save the results
    write_results(results, batch_size, triple_part_to_train_on)

## Idea explained (for writing): ##
# This incremental approach allows you to:
# Build a Solid Foundation: By focusing on predicting the subject, predicate, or object of a knowledge graph triple, you're essentially testing the ability of the model to learn meaningful patterns from the data without the complexity of generating or handling negative samples. It's a good way to understand how well the model can identify the relationships within the data.
# Gain Insight into Triple Representation: Predicting individual parts of the triple can help you understand how well the model handles various aspects of the data. Since predicates typically have fewer unique labels than subjects or objects, starting with predicate prediction gives a manageable starting point.
# Create Better Features: Through this process, you'll learn what kind of features (tokenization, embeddings, etc.) work best for each part of the triple. This knowledge can later help you design more efficient and meaningful embeddings when you move on to full triple prediction.
# Prepare for Negative Samples: Once you've successfully predicted the individual parts of the triples, the next step (introducing negative samples for link prediction) will seem more straightforward. You'll have already identified the important features and relationships, and the model will only need to learn to distinguish valid triples from invalid ones.
# Ensure Model Robustness: Predicting individual parts will show if your model generalizes well and can identify key relationships. Once you're confident in its performance, you can extend it to more complex tasks like link prediction, where differentiating between positive and negative samples becomes crucial.