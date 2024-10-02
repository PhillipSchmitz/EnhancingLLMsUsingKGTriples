import random
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails):
        # Embeddings for head, relation, and tail
        head_embed = self.entity_embeddings(heads)
        relation_embed = self.relation_embeddings(relations)
        tail_embed = self.entity_embeddings(tails)

        # Compute distance score (L2 norm)
        score = torch.norm(head_embed + relation_embed - tail_embed, p=2, dim=1)
        return -score  # Return negative distance as score (for binary classification)


class ComplEX(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ComplEX, self).__init__()
        self.entity_embeddings_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embeddings_real.weight)
        nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_real.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_imag.weight)

    def forward(self, heads, relations, tails):
        # Embeddings for real and imaginary parts
        head_real = self.entity_embeddings_real(heads)
        head_imag = self.entity_embeddings_imag(heads)
        relation_real = self.relation_embeddings_real(relations)
        relation_imag = self.relation_embeddings_imag(relations)
        tail_real = self.entity_embeddings_real(tails)
        tail_imag = self.entity_embeddings_imag(tails)

        # ComplEX scoring function
        score_real = torch.sum(head_real * relation_real * tail_real + head_imag * relation_imag * tail_real, dim=1)
        score_imag = torch.sum(head_real * relation_imag * tail_imag - head_imag * relation_real * tail_imag, dim=1)

        # Combine real and imaginary scores to get the final score
        score = score_real + score_imag
        return score


class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RotatE, self).__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even for RotatE")

        self.embedding_dim = embedding_dim // 2
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.phase_relation = nn.Embedding(num_relations, self.embedding_dim)

        nn.init.uniform_(self.entity_embeddings.weight, -6 / self.embedding_dim ** 0.5, 6 / self.embedding_dim ** 0.5)
        nn.init.uniform_(self.phase_relation.weight, -3.14159, 3.14159)

    def forward(self, heads, relations, tails):
        # Embeddings for head, relation, and tail
        head_embed = self.entity_embeddings(heads)
        tail_embed = self.entity_embeddings(tails)
        phase_relation = self.phase_relation(relations)

        # Split real and imaginary parts
        re_head, im_head = torch.chunk(head_embed, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail_embed, 2, dim=1)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # Compute real and imaginary scores
        re_score = re_head * re_relation - im_head * im_relation - re_tail
        im_score = re_head * im_relation + im_head * re_relation - im_tail

        # Combine real and imaginary scores
        score = torch.sum(re_score ** 2 + im_score ** 2, dim=1)
        return -score  # Return negative score (similar to TransE)

# Define the Classification Model
class ClassificationModel(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, model_name='TransE'):
        super(ClassificationModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.model_name = model_name

        # Define embeddings for entities and relations
        self.entity_embeddings = nn.Embedding(entity_count, embedding_dim)
        self.relation_embeddings = nn.Embedding(relation_count, embedding_dim)

        # Classifier head for binary classification (1 output for valid/invalid)
        if model_name == 'ComplEX':
            self.entity_embeddings_real = nn.Embedding(entity_count, embedding_dim)
            self.entity_embeddings_imag = nn.Embedding(entity_count, embedding_dim)
            self.relation_embeddings_real = nn.Embedding(relation_count, embedding_dim)
            self.relation_embeddings_imag = nn.Embedding(relation_count, embedding_dim)
            self.classifier = nn.Linear(embedding_dim * 8, 1)  # 8 features total
        elif model_name == 'RotatE':
            if embedding_dim % 2 != 0:
                raise ValueError("Embedding dimension must be even for RotatE")
            self.embedding_dim = embedding_dim // 2  # Halve the embedding dim
            self.phase_relation = nn.Embedding(relation_count, self.embedding_dim)
            self.classifier = nn.Linear(embedding_dim * 1, 1)  # 2 * (embedding_dim // 2) = embedding_dim
        else:  # Default to TransE
            self.classifier = nn.Linear(embedding_dim * 2, 1)  # Output single score for binary classification

    def forward(self, heads, relations, tails):
        head_embeds = self.entity_embeddings(heads)
        tail_embeds = self.entity_embeddings(tails)
        relation_embeds = self.relation_embeddings(relations)

        # Scoring for each model type
        if self.model_name == 'TransE':
            scores = torch.cat([head_embeds + relation_embeds - tail_embeds, head_embeds - tail_embeds], dim=1)
        elif self.model_name == 'ComplEX':
            head_real = self.entity_embeddings_real(heads)
            head_imag = self.entity_embeddings_imag(heads)
            relation_real = self.relation_embeddings_real(relations)
            relation_imag = self.relation_embeddings_imag(relations)
            tail_real = self.entity_embeddings_real(tails)
            tail_imag = self.entity_embeddings_imag(tails)

            score_real = head_real * relation_real * tail_real + head_imag * relation_imag * tail_real
            score_imag = head_real * relation_imag * tail_imag - head_imag * relation_real * tail_imag

            scores = torch.cat([score_real, score_imag, head_real, head_imag, relation_real, relation_imag, tail_real, tail_imag], dim=1)
        elif self.model_name == 'RotatE':
            head_embed = self.entity_embeddings(heads)
            tail_embed = self.entity_embeddings(tails)
            phase_relation = self.phase_relation(relations)

            re_head, im_head = torch.chunk(head_embed, 2, dim=1)
            re_tail, im_tail = torch.chunk(tail_embed, 2, dim=1)
            re_relation = torch.cos(phase_relation)
            im_relation = torch.sin(phase_relation)

            # Calculate scores
            re_score = re_head * re_relation - im_head * im_relation - re_tail
            im_score = re_head * im_relation + im_head * re_relation - im_tail

            # Concatenate scores
            scores = torch.cat([re_score, im_score], dim=1)  # Output shape will be (batch_size, 100)

        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        # Classifier output (single score for binary classification)
        logits = self.classifier(scores)
        return logits.view(-1)  # Output is a single scalar per triple.


# Function to create entity embeddings
def create_entity_embeddings(triples_df, model_name='TransE'):
    entities = set(triples_df['Subject']).union(set(triples_df['Object']))
    relations = set(triples_df['Predicate'])

    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {relation: idx for idx, relation in enumerate(relations)}

    return entity2id, relation2id

def save_model_and_embeddings(model, save_dir='models/', model_name='TransE'):
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name.lower()}_model.pth'))

    # Save the entity and relation embeddings based on the model type
    if model_name == 'TransE':
        torch.save({
            'entity_embeddings': model.entity_embeddings.weight.detach().cpu(),
            'relation_embeddings': model.relation_embeddings.weight.detach().cpu()
        }, os.path.join(save_dir, f'{model_name.lower()}_embeddings.pth'))

    elif model_name == 'ComplEX':
        torch.save({
            'entity_embeddings_real': model.entity_embeddings_real.weight.detach().cpu(),
            'entity_embeddings_imag': model.entity_embeddings_imag.weight.detach().cpu(),
            'relation_embeddings_real': model.relation_embeddings_real.weight.detach().cpu(),
            'relation_embeddings_imag': model.relation_embeddings_imag.weight.detach().cpu()
        }, os.path.join(save_dir, f'{model_name.lower()}_embeddings.pth'))

    elif model_name == 'RotatE':
        torch.save({
            'entity_embeddings': model.entity_embeddings.weight.detach().cpu(),
            'phase_relation': model.phase_relation.weight.detach().cpu()
        }, os.path.join(save_dir, f'{model_name.lower()}_embeddings.pth'))

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"Model and embeddings saved in {save_dir}")

def load_model_and_embeddings(num_entities, num_relations, embedding_dim, model_name='TransE', load_dir='models/'):
    """
    Load a saved ClassificationModel and its embeddings based on the model type.
    :param num_entities: Number of entities in the knowledge graph.
    :param num_relations: Number of relations in the knowledge graph.
    :param embedding_dim: Dimensionality of embeddings.
    :param model_name: The name of the model ('TransE', 'ComplEX', or 'RotatE').
    :param load_dir: Directory where the model and embeddings are saved.
    :return: Loaded ClassificationModel with embeddings.
    """
    # Initialize the ClassificationModel with the classifier
    model = ClassificationModel(num_entities, num_relations, embedding_dim, model_name=model_name)

    # Load model state (set strict=False to allow loading mismatching keys)
    model.load_state_dict(torch.load(os.path.join(load_dir, f'{model_name.lower()}_model.pth'), weights_only=True))

    # Load the embeddings
    embeddings = torch.load(os.path.join(load_dir, f'{model_name.lower()}_embeddings.pth'), weights_only=True)

    # Load the embeddings into the model
    if model_name == 'TransE':
        model.entity_embeddings.weight.data.copy_(embeddings['entity_embeddings'])
        model.relation_embeddings.weight.data.copy_(embeddings['relation_embeddings'])
    elif model_name == 'ComplEX':
        model.entity_embeddings_real.weight.data.copy_(embeddings['entity_embeddings_real'])
        model.entity_embeddings_imag.weight.data.copy_(embeddings['entity_embeddings_imag'])
        model.relation_embeddings_real.weight.data.copy_(embeddings['relation_embeddings_real'])
        model.relation_embeddings_imag.weight.data.copy_(embeddings['relation_embeddings_imag'])
    elif model_name == 'RotatE':
        model.entity_embeddings.weight.data.copy_(embeddings['entity_embeddings'])
        model.phase_relation.weight.data.copy_(embeddings['phase_relation'])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"Model and embeddings loaded from {load_dir}")

    return model

# Function to get batches for training
def get_batch(df, batch_size):
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        yield df.iloc[start:end]

def generate_negative_samples(batch, entity_list, num_negatives=1):
    """
    Generate negative samples by corrupting either the head or tail of the triples.
    The number of negative samples per triple can be adjusted using num_negatives.
    """
    negative_samples = []

    for _, row in batch.iterrows():
        # For each triple in the batch
        for _ in range(num_negatives):
            corrupt_head = random.choice([True, False])  # Randomly decide whether to corrupt head or tail

            if corrupt_head:
                # Corrupt the head entity (replace with a random entity)
                corrupt_entity = random.choice(list(entity_list))
                negative_samples.append({
                    'Subject': corrupt_entity,
                    'Predicate': row['Predicate'],
                    'Object': row['Object']
                })
            else:
                # Corrupt the tail entity (replace with a random entity)
                corrupt_entity = random.choice(list(entity_list))
                negative_samples.append({
                    'Subject': row['Subject'],
                    'Predicate': row['Predicate'],
                    'Object': corrupt_entity
                })

    # Convert negative samples into a pandas DataFrame for consistency with batch format
    return pd.DataFrame(negative_samples)

# Training function for classification
def train_triple_classification_model_batch(model, triples_df, entity2id, relation2id, model_name="TransE", epochs=100, learning_rate=0.001, batch_size=128, num_negatives=1, log_file="training_log.txt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Open a file to log training information
    with open(log_file, "w") as f:
        f.write(f"Training: {model_name} model\n")
        f.write(f"Epochs: {epochs}, Learning rate: {learning_rate}, Batch size: {batch_size}\n\n")
        f.write("Epoch\tLoss\n")

        for epoch in range(epochs):
            total_loss = 0

            # Iterate over batches
            for i, batch in enumerate(get_batch(triples_df, batch_size)):
                # Prepare positive samples (forward pass for positive samples)
                heads = torch.tensor([entity2id[h] for h in batch['Subject']], dtype=torch.long)
                relations = torch.tensor([relation2id[r] for r in batch['Predicate']], dtype=torch.long)
                tails = torch.tensor([entity2id[t] for t in batch['Object']], dtype=torch.long)

                # Positive scores (these should be the logits of the model for the positive triples)
                pos_scores = model(heads, relations, tails)

                # Generate negative samples with the same batch size
                neg_batch = generate_negative_samples(batch, entity_list=entity2id, num_negatives=num_negatives)
                neg_heads = torch.tensor([entity2id[h] for h in neg_batch['Subject']], dtype=torch.long)
                neg_relations = torch.tensor([relation2id[r] for r in neg_batch['Predicate']], dtype=torch.long)
                neg_tails = torch.tensor([entity2id[t] for t in neg_batch['Object']], dtype=torch.long)

                # Negative scores (these should be the logits of the model for the negative triples)
                neg_scores = model(neg_heads, neg_relations, neg_tails)

                # Labels: Positive triples (1), Negative triples (0)
                pos_labels = torch.ones(pos_scores.size(0), dtype=torch.float32)
                neg_labels = torch.zeros(neg_scores.size(0), dtype=torch.float32)

                # Combine positive and negative scores and labels
                all_scores = torch.cat([pos_scores, neg_scores], dim=0)  # Combined logits for positive and negative triples
                all_labels = torch.cat([pos_labels, neg_labels], dim=0)  # Combined labels (1 for positive, 0 for negative)

                # Compute binary classification loss (BCEWithLogitsLoss)
                loss = loss_fn(all_scores.view(-1), all_labels)  # Reshape to make sure it's the right size

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Log training progress
            f.write(f"{epoch + 1}\t{total_loss:.4f}\n")
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    print(f"Training complete! Model and log saved in {log_file}")

def predict_triple_validity(model, subject, predicate, obj, entity2id, relation2id, threshold=0.5):
    """
    Predict whether a given triple (subject, predicate, object) is valid (positive) or not (negative).

    :param model: The trained classification model.
    :param subject: The subject of the triple.
    :param predicate: The predicate of the triple.
    :param obj: The object of the triple.
    :param entity2id: Dictionary mapping entities to their IDs.
    :param relation2id: Dictionary mapping relations to their IDs.
    :param threshold: The threshold for classifying the probability as positive.
    :return: Prediction of whether the triple is valid (positive) or not (negative).
    """
    model.eval()

    # Convert input to IDs
    if subject not in entity2id or obj not in entity2id or predicate not in relation2id:
        raise ValueError("Subject, predicate, or object not found in the entity or relation mappings.")

    head_id = torch.tensor([entity2id[subject]], dtype=torch.long)
    relation_id = torch.tensor([relation2id[predicate]], dtype=torch.long)
    tail_id = torch.tensor([entity2id[obj]], dtype=torch.long)

    with torch.no_grad():
        # Forward pass
        logits = model(head_id, relation_id, tail_id)

        # Apply sigmoid to get probabilities
        prob = torch.sigmoid(logits).item()

    # Predict whether the triple is positive or negative
    prediction = prob >= threshold

    return "valid" if prediction else "invalid"


def combine_positive_negative_samples(positive_df, negative_df):
    # Assign labels
    positive_df['Label'] = 1  # Label positive samples as 1
    negative_df['Label'] = 0  # Label negative samples as 0

    # Combine positive and negative samples
    mixed_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Shuffle the dataset to mix positive and negative samples
    mixed_df = mixed_df.sample(frac=1).reset_index(drop=True)

    return mixed_df


# Evaluation function for binary classification model
def evaluate_classification_model(model, mixed_df, entity2id, relation2id, model_name, log_file="evaluation_log.txt"):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in get_batch(mixed_df, batch_size=128):
            heads = torch.tensor([entity2id[h] for h in batch['Subject']], dtype=torch.long)
            relations = torch.tensor([relation2id[r] for r in batch['Predicate']], dtype=torch.long)
            tails = torch.tensor([entity2id[t] for t in batch['Object']], dtype=torch.long)

            # Forward pass
            logits = model(heads, relations, tails)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).view(-1)

            # Convert probabilities to binary predictions (0 or 1)
            preds = (probs >= 0.5).long().tolist()

            # Actual labels (0 for negative samples, 1 for positive samples)
            targets = batch['Label'].tolist()

            all_preds.extend(preds)
            all_targets.extend(targets)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    print(f"Model: {model_name}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Write metrics to a file
    with open(log_file, 'w') as f:
        f.write(f"Evaluation: {model_name} model\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"Evaluation metrics saved to {log_file}")

# Test function for binary classification model
def test_classification_model(model, mixed_df, entity2id, relation2id, model_name, log_file="test_log.txt"):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in get_batch(mixed_df, batch_size=128):
            heads = torch.tensor([entity2id[h] for h in batch['Subject']], dtype=torch.long)
            relations = torch.tensor([relation2id[r] for r in batch['Predicate']], dtype=torch.long)
            tails = torch.tensor([entity2id[t] for t in batch['Object']], dtype=torch.long)

            # Forward pass
            logits = model(heads, relations, tails)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).view(-1)

            # Convert probabilities to binary predictions (0 or 1)
            preds = (probs >= 0.5).long().tolist()

            # Actual labels (0 for negative samples, 1 for positive samples)
            targets = batch['Label'].tolist()

            all_preds.extend(preds)
            all_targets.extend(targets)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    print(f"Model: {model_name}, Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Write metrics to a file
    with open(log_file, 'w') as f:
        f.write(f"Test: {model_name} model\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"Test metrics saved to {log_file}")

# Main execution
if __name__ == "__main__":
    # Read the text file into a DataFrame
    df_train = pd.read_csv('../data/train.txt', sep='\t', header=None, names=['Subject', 'Predicate', 'Object'])
    df_train.to_csv('train.csv', index=False)

    df_valid = pd.read_csv('../data/valid.txt', sep='\t', header=None, names=['Subject', 'Predicate', 'Object'])
    df_valid.to_csv('valid.csv', index=False)
    df_valid_negatives = pd.read_csv('../data/valid_negatives.txt', sep='\t', header=None, names=['Subject', 'Predicate', 'Object'])
    df_valid_negatives.to_csv('valid_negatives.csv', index=False)

    df_test = pd.read_csv('../data/test.txt', sep='\t', header=None, names=['Subject', 'Predicate', 'Object'])
    df_test.to_csv('test.csv', index=False)
    df_test_negatives = pd.read_csv('../data/test_negatives.txt', sep='\t', header=None, names=['Subject', 'Predicate', 'Object'])
    df_test_negatives.to_csv('test_negatives.csv', index=False)

    # Load the training and test datasets
    triples_df_train = pd.read_csv('train.csv')
    triples_df_valid = pd.read_csv('valid.csv')
    triples_df_valid_negatives = pd.read_csv('valid_negatives.csv')
    triples_df_test = pd.read_csv('test.csv')
    triples_df_test_negatives = pd.read_csv('test_negatives.csv')

    # Pick the model to use
    model_name = 'TransE'  # You can choose between 'TransE' (default), 'ComplEX', 'RotatE'

    # Define the directory for saving metrics
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Create entity embeddings and initialize the classification model
    entity2id, relation2id = create_entity_embeddings(triples_df_train, model_name=model_name)
    classification_model = ClassificationModel(len(entity2id), len(relation2id), embedding_dim=100, model_name=model_name)

    # Train the classification model
    train_triple_classification_model_batch(
        classification_model,
        triples_df_train,
        entity2id,
        relation2id,
        model_name=model_name,
        epochs=100,
        learning_rate=0.001,
        batch_size=128,
        num_negatives=1,
        log_file=os.path.join(metrics_dir, f"{model_name}_training_log.txt")
    )

    # Optionally, save the model and embeddings
    save_model_and_embeddings(classification_model, save_dir='models/', model_name=model_name)

    # Combine positive and negative validation samples into a single mixed dataframe
    validation_df_mixed = combine_positive_negative_samples(triples_df_valid, triples_df_valid_negatives)

    # Evaluate the classification model and save metrics to a file
    evaluate_classification_model(
        classification_model,
        validation_df_mixed,
        entity2id,
        relation2id,
        model_name=model_name,
        log_file=os.path.join(metrics_dir, f"{model_name}_evaluation_log.txt")
    )

    # Combine positive and negative test samples into a single mixed dataframe
    test_df_mixed = combine_positive_negative_samples(triples_df_test, triples_df_test_negatives)

    # Test the classification model and save metrics to a file
    test_classification_model(
        classification_model,
        test_df_mixed,
        entity2id,
        relation2id,
        model_name,
        log_file=os.path.join(metrics_dir, f"{model_name}_test_log.txt")
    )

    # Optionally, predict the validity of a specific triple
    #subject = 'Entity1'
    #predicate = 'Predicate1'
    #object = 'Entity2'
    #validity = predict_triple_validity(classification_model,subject,predicate,object,entity2id,relation2id)
    #print(f"Triple ({subject}, {predicate}, {object}) is predicted to be: {validity}")

    # Optionally, load the model and embeddings and re-evaluate the loaded model
    #loaded_model = load_model_and_embeddings(num_entities=len(entity2id),num_relations=len(relation2id),embedding_dim=100,model_name=model_name,load_dir='models/')
    #evaluate_classification_model(loaded_model, validation_df_mixed, entity2id, relation2id, model_name=model_name, log_file=os.path.join(metrics_dir, f"loaded_{model_name}_evaluation_log.txt"))
    #test_classification_model(loaded_model, test_df_mixed, entity2id, relation2id, model_name=model_name, log_file=os.path.join(metrics_dir, f"loaded_{model_name}_test_log.txt"))

    # Optionally, predict the validity of a specific triple using the loaded model
    #validity_loaded = predict_triple_validity(classification_model, subject, predicate, object, entity2id, relation2id)
    #print(f"Triple ({subject}, {predicate}, {object}) is predicted to be: {validity_loaded}")