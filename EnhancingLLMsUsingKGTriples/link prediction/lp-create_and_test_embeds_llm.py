import random
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


# TransE model
# Strengths: Simple and efficient, works well for datasets with hierarchical relationships.
# Weaknesses: Struggles with complex relationships like one-to-many, many-to-one, and many-to-many.
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(self, head_id, relation_id, tail_id):
        head_embed = self.entity_embeddings(head_id)
        relation_embed = self.relation_embeddings(relation_id)
        tail_embed = self.entity_embeddings(tail_id)

        # TransE scoring: ||head + relation - tail|| (L2 norm)
        score = torch.norm(head_embed + relation_embed - tail_embed, p=2, dim=-1)
        return score

    def forward(self, heads, relations, tails):
        head_embed = self.entity_embeddings(heads)
        relation_embed = self.relation_embeddings(relations)
        tail_embed = self.entity_embeddings(tails)

        return torch.norm(head_embed + relation_embed - tail_embed, p=2, dim=1)


# Analysis:
# 1.) The dataset contains a variety of relationships between entities, i.e. predicates)
# 2.) We need to handle complex and asymmetric relationships better (-> ComplEX and RotatE models)

# ComplEX model
# Strengths: Handles asymmetric and complex relationships well due to its use of complex-valued embeddings.
# Weaknesses: More computationally intensive compared to TransE.
class ComplEX(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ComplEX, self).__init__()
        self.entity_embeddings_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(num_relations, embedding_dim)

    def forward(self, heads, relations, tails):
        head_real = self.entity_embeddings_real(heads)
        head_imag = self.entity_embeddings_imag(heads)
        relation_real = self.relation_embeddings_real(relations)
        relation_imag = self.relation_embeddings_imag(relations)
        tail_real = self.entity_embeddings_real(tails)
        tail_imag = self.entity_embeddings_imag(tails)

        score_real = head_real * relation_real * tail_real + head_imag * relation_imag * tail_real
        score_imag = head_real * relation_imag * tail_imag - head_imag * relation_real * tail_imag

        score = score_real.sum(dim=1) + score_imag.sum(dim=1)
        return score

    def get_entity_embedding(self, entity_id):
        entity_id = torch.tensor([entity_id], dtype=torch.long)  # Convert entity_id to tensor
        real_part = self.entity_embeddings_real(entity_id)
        imag_part = self.entity_embeddings_imag(entity_id)
        return real_part, imag_part

    def get_relation_embedding(self, relation_id):
        relation_id = torch.tensor([relation_id], dtype=torch.long)  # Convert relation_id to tensor
        real_part = self.relation_embeddings_real(relation_id)
        imag_part = self.relation_embeddings_imag(relation_id)
        return real_part, imag_part

    def score(self, head_embed, relation_embed, tail_embed):
        # Unpack real and imaginary parts
        head_real, head_imag = head_embed
        relation_real, relation_imag = relation_embed
        tail_real, tail_imag = tail_embed

        # Compute the real and imaginary components of the score
        score_real = head_real * relation_real * tail_real + head_imag * relation_imag * tail_real
        score_imag = head_real * relation_imag * tail_imag - head_imag * relation_real * tail_imag

        # Sum the scores along the embedding dimension (dim=-1), not across the entire batch
        score = torch.sum(score_real + score_imag, dim=-1)

        return score


# RotatE model
# Strengths: Handles both symmetric and asymmetric relationships effectively by rotating entity embeddings in complex space.
# Weaknesses: More complex and computationally intensive.
class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RotatE, self).__init__()

        # Ensure embedding_dim is even, as we will split it into real and imaginary parts
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even for RotatE")

        self.embedding_dim = embedding_dim // 2  # Use half for real, half for imaginary
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.phase_relation = nn.Embedding(num_relations, self.embedding_dim)

        # Initialize embeddings
        nn.init.uniform_(self.entity_embeddings.weight, -6 / self.embedding_dim ** 0.5, 6 / self.embedding_dim ** 0.5)
        nn.init.uniform_(self.phase_relation.weight, -3.14159, 3.14159)

    def forward(self, heads, relations, tails):
        # Get entity and relation embeddings
        head_embed = self.entity_embeddings(heads)
        tail_embed = self.entity_embeddings(tails)
        phase_relation = self.phase_relation(relations)

        # Split entity embeddings into real and imaginary parts
        re_head, im_head = torch.chunk(head_embed, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail_embed, 2, dim=1)

        # Compute real and imaginary parts for the relation embedding
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # Compute the score
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        re_score = re_score - re_tail
        im_score = im_score - im_tail

        # Calculate the final score
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = score.sum(dim=1)

        return score

    def get_entity_embedding(self, entity_id):
        entity_id = torch.tensor([entity_id], dtype=torch.long)
        return self.entity_embeddings(entity_id)

    def get_relation_embedding(self, relation_id):
        relation_id = torch.tensor([relation_id], dtype=torch.long)
        return self.phase_relation(relation_id)

    def score(self, head_embed, relation_embed, tail_embed):
        # Split head and tail embeddings into real and imaginary parts
        re_head, im_head = torch.chunk(head_embed, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail_embed, 2, dim=1)

        # Apply rotation to head embeddings using relation embeddings (as rotations)
        re_relation, im_relation = torch.cos(relation_embed), torch.sin(relation_embed)

        # Rotate the head embedding by the relation
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        # Calculate the difference between the rotated head and tail embeddings
        re_diff = re_score - re_tail
        im_diff = im_score - im_tail

        # Stack the real and imaginary differences and compute the norm (L2 distance)
        score = torch.stack([re_diff, im_diff], dim=0)
        score = torch.norm(score, p=2, dim=0)

        # Sum the distances for each triple
        score = torch.sum(score, dim=1)

        return -score  # Negative distance, higher score for closer embeddings


def create_entity_embeddings(df, model_name='TransE'):
    # Create a set of unique entities
    entities = set(df['Subject']).union(set(df['Object']))

    # Create entity and relation dictionaries
    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {relation: idx for idx, relation in enumerate(df['Predicate'].unique())}

    # Define hyperparameters
    embedding_dim = 100
    margin = 1.0
    learning_rate = 0.001
    epochs = 100

    # Initialize the model based on model_name
    if model_name == 'TransE':
        model = TransE(len(entities), len(relation2id), embedding_dim)
    elif model_name == 'ComplEX':
        model = ComplEX(len(entities), len(relation2id), embedding_dim)
    elif model_name == 'RotatE':
        model = RotatE(len(entities), len(relation2id), embedding_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert DataFrame columns to tensors
    heads = torch.tensor(df['Subject'].map(entity2id).values)
    relations = torch.tensor(df['Predicate'].map(relation2id).values)
    tails = torch.tensor(df['Object'].map(entity2id).values)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        pos_scores = model(heads, relations, tails)
        neg_tails = torch.randint(len(entities), (len(df),))
        neg_scores = model(heads, relations, neg_tails)
        loss = torch.sum(torch.max(pos_scores - neg_scores + margin, torch.zeros_like(pos_scores)))
        loss.backward()
        optimizer.step()

    return model, entity2id, relation2id


def save_model_and_embeddings(model, save_dir='models/', model_name='TransE'):
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name.lower()}_model.pth'))

    # Save the entity and relation embeddings
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
    Load a saved model and its embeddings based on the model type.
    :param num_entities: Number of entities in the knowledge graph.
    :param num_relations: Number of relations in the knowledge graph.
    :param embedding_dim: Dimensionality of embeddings.
    :param model_name: The name of the model ('TransE', 'ComplEX', or 'RotatE').
    :param load_dir: Directory where the model and embeddings are saved.
    :return: Loaded model with embeddings.
    """

    if model_name == 'TransE':
        # Initialize TransE model
        model = TransE(num_entities, num_relations, embedding_dim)

        # Load model state (with weights_only=True to suppress warning)
        model.load_state_dict(torch.load(os.path.join(load_dir, 'transe_model.pth'), weights_only=True))

        # Load the embeddings (with weights_only=True to suppress warning)
        embeddings = torch.load(os.path.join(load_dir, 'transe_embeddings.pth'), weights_only=True)
        model.entity_embeddings.weight.data.copy_(embeddings['entity_embeddings'])
        model.relation_embeddings.weight.data.copy_(embeddings['relation_embeddings'])

    elif model_name == 'ComplEX':
        # Initialize ComplEX model
        model = ComplEX(num_entities, num_relations, embedding_dim)

        # Load model state (with weights_only=True)
        model.load_state_dict(torch.load(os.path.join(load_dir, 'complex_model.pth'), weights_only=True))

        # Load the embeddings (with weights_only=True)
        embeddings = torch.load(os.path.join(load_dir, 'complex_embeddings.pth'), weights_only=True)
        model.entity_embeddings_real.weight.data.copy_(embeddings['entity_embeddings_real'])
        model.entity_embeddings_imag.weight.data.copy_(embeddings['entity_embeddings_imag'])
        model.relation_embeddings_real.weight.data.copy_(embeddings['relation_embeddings_real'])
        model.relation_embeddings_imag.weight.data.copy_(embeddings['relation_embeddings_imag'])

    elif model_name == 'RotatE':
        # Initialize RotatE model
        model = RotatE(num_entities, num_relations, embedding_dim)

        # Load model state (with weights_only=True)
        model.load_state_dict(torch.load(os.path.join(load_dir, 'rotate_model.pth'), weights_only=True))

        # Load the embeddings (with weights_only=True)
        embeddings = torch.load(os.path.join(load_dir, 'rotate_embeddings.pth'), weights_only=True)
        model.entity_embeddings.weight.data.copy_(embeddings['entity_embeddings'])
        model.phase_relation.weight.data.copy_(embeddings['phase_relation'])

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"Model and embeddings loaded from {load_dir}")

    return model


def get_batch(triples_df, batch_size):
    """
    Generator function to yield batches from the triples dataframe.
    :param triples_df: Pandas DataFrame containing the triples (Subject, Predicate, Object).
    :param batch_size: Number of triples per batch.
    :yield: A batch of triples as a pandas DataFrame.
    """
    num_batches = len(triples_df) // batch_size + (1 if len(triples_df) % batch_size != 0 else 0)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(triples_df))
        yield triples_df.iloc[start_idx:end_idx]


def generate_negative_sample(batch, entity2id, corrupt_head=True):
    """
    Generate negative samples by corrupting the head or tail of the positive triples.
    The number of negative samples will match the batch size.
    :param batch: A batch of triples (Subject, Predicate, Object).
    :param entity2id: Dictionary mapping entities to their IDs.
    :param corrupt_head: If True, corrupt the head, otherwise corrupt the tail.
    :return: Corrupted head or tail.
    """
    corrupted_triples = []

    for index, row in batch.iterrows():
        head, relation, tail = row['Subject'], row['Predicate'], row['Object']
        corrupted_entity = random.choice(list(entity2id.keys()))

        if corrupt_head:
            corrupted_triples.append((corrupted_entity, relation, tail))
        else:
            corrupted_triples.append((head, relation, corrupted_entity))

    return pd.DataFrame(corrupted_triples, columns=['Subject', 'Predicate', 'Object'])


# TransE uses margin-based ranking loss to enforce a separation between positive and negative triples.
# This is because TransE's scoring is based on distance (L2 norm), and the ranking loss works by
# ensuring positive triples have lower distances (scores) than negative ones by at least a margin.
def margin_ranking_loss(pos_score, neg_score, margin=1.0):
    """
    Margin-based ranking loss function.
    :param pos_score: Score for the positive (valid) triple.
    :param neg_score: Score for the negative (corrupted) triple.
    :param margin: Margin value to enforce distance between positive and negative scores.
    :return: Loss value.
    """
    return torch.relu(pos_score - neg_score + margin).mean()


# ComplEX benefits from binary cross-entropy because it uses complex-valued embeddings.
# Binary cross-entropy helps optimize the probabilistic nature of the scoring function in ComplEX,
# which deals with real and imaginary components, making it a better fit for handling positive and negative examples.
def binary_cross_entropy_loss(pos_score, neg_score):
    pos_loss = -torch.log(torch.sigmoid(pos_score))
    neg_loss = -torch.log(torch.sigmoid(-neg_score))
    return (pos_loss + neg_loss).mean()


# RotatE works in a complex space with rotations in a unit circle, and it is commonly trained with self-adversarial loss.
# The adversarial loss applies softmax to negative samples, weighting harder negatives more,
# which fits well with RotatE's ability to distinguish between complex relational structures.
def adversarial_loss(pos_score, neg_score, alpha=0.5):
    """
    Self-adversarial loss used for RotatE.
    :param pos_score: Score for positive triples.
    :param neg_score: Score for negative triples.
    :param alpha: Temperature parameter controlling the softmax.
    """
    pos_loss = -torch.log(torch.sigmoid(pos_score))
    neg_weights = torch.softmax(neg_score * alpha, dim=-1)
    neg_loss = -torch.sum(neg_weights * torch.log(torch.sigmoid(-neg_score)), dim=-1)
    return (pos_loss + neg_loss).mean()


def train_model_batch(model, triples_df_train, entity2id, relation2id, model_name="TransE", epochs=100,
                      learning_rate=0.001,
                      margin=1.0, batch_size=128, log_file="training_log.txt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Open a file to log training information
    with open(log_file, "w") as f:
        f.write(f"Training {model_name} model\n")
        f.write(f"Epochs: {epochs}, Learning rate: {learning_rate}, Margin: {margin}, Batch size: {batch_size}\n\n")
        f.write("Epoch\tLoss\n")

        for epoch in range(epochs):
            total_loss = 0
            for i, batch in enumerate(get_batch(triples_df_train, batch_size)):
                heads = torch.tensor([entity2id[h] for h in batch['Subject']], dtype=torch.long)
                relations = torch.tensor([relation2id[r] for r in batch['Predicate']], dtype=torch.long)
                tails = torch.tensor([entity2id[t] for t in batch['Object']], dtype=torch.long)

                # Forward pass for positive samples
                pos_scores = model(heads, relations, tails)

                # Generate negative samples with the same batch size
                neg_batch = generate_negative_sample(batch, entity2id,
                                                     corrupt_head=random.choice([True, False]))  # Corrupting heads
                neg_heads = torch.tensor([entity2id[h] for h in neg_batch['Subject']], dtype=torch.long)
                neg_relations = torch.tensor([relation2id[r] for r in neg_batch['Predicate']], dtype=torch.long)
                neg_tails = torch.tensor([entity2id[t] for t in neg_batch['Object']], dtype=torch.long)

                neg_scores = model(neg_heads, neg_relations, neg_tails)

                # Choose loss function based on model type
                if model_name == "TransE":
                    loss = margin_ranking_loss(pos_scores, neg_scores, margin=margin)
                elif model_name == "ComplEX":
                    loss = binary_cross_entropy_loss(pos_scores, neg_scores)
                elif model_name == "RotatE":
                    loss = adversarial_loss(pos_scores, neg_scores)
                else:
                    raise ValueError(f"Unknown model name: {model_name}")

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Log training progress
            f.write(f"{epoch + 1}\t{total_loss:.4f}\n")
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    print(f"Training complete! Model and log saved in {log_file}")


def predict_tail(model, head, relation, entity2id, relation2id, model_name='TransE'):
    head_id = entity2id[head]
    relation_id = relation2id[relation]

    if model_name == 'TransE':
        head_embed = model.entity_embeddings(torch.tensor([head_id]).long())
        relation_embed = model.relation_embeddings(torch.tensor([relation_id]).long())

    elif model_name == 'ComplEX' or model_name == 'RotatE':
        head_embed = model.get_entity_embedding(head_id)
        relation_embed = model.get_relation_embedding(relation_id)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    scores = []
    for tail_id in range(len(entity2id)):
        if model_name == 'TransE':
            tail_embed = model.entity_embeddings(torch.tensor([tail_id]).long())
            score = torch.norm(head_embed + relation_embed - tail_embed, p=1).item()

        elif model_name == 'ComplEX' or model_name == 'RotatE':
            tail_embed = model.get_entity_embedding(tail_id)
            score = model.score(head_embed, relation_embed, tail_embed).item()

        scores.append(score)

    # Get the tail entity with the highest score
    predicted_tail_id = scores.index(min(scores))  # Assuming lower score is better
    predicted_tail = list(entity2id.keys())[list(entity2id.values()).index(predicted_tail_id)]

    return predicted_tail

def combine_positive_negative_samples(positive_df, negative_df):
    # Add label column: 1 for positive samples, 0 for negative samples
    positive_df['Label'] = 1
    negative_df['Label'] = 0

    # Combine positive and negative samples
    mixed_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Shuffle the dataset to mix positive and negative samples
    mixed_df = mixed_df.sample(frac=1).reset_index(drop=True)

    return mixed_df

def evaluate_model(model, mixed_df, entity2id, relation2id, model_name, log_file):
    """
    Function to evaluate the trained model and return metrics like Mean Rank, MRR, and Hits@10.
    This is a placeholder. The actual function should perform evaluation on the validation/test data.
    """
    # Determine the maximum number of CPU cores available on the machine.
    max_processes = os.cpu_count()

    mean_rank, mrr = evaluate_ranking_parallel(model, mixed_df, entity2id, relation2id,
                                               num_processes=max_processes, model_name=model_name)
    print(f"Mean Rank: {mean_rank}")  # (1: optimal)
    print(f"MRR: {mrr}")  # (1: optimal)

    hits_at_10 = hits_at_k_parallel(model, mixed_df, entity2id, relation2id, k=10, num_processes=max_processes,
                                    model_name=model_name)
    print(f"Hits@10: {hits_at_10}")  # (optimal: 1th rank out of all k ranks)

    mse_value = mean_squared_error_parallel(model, mixed_df, entity2id, relation2id, num_processes=max_processes,
                                            model_name=model_name)
    print(f"Mean Squared Error: {mse_value}")  # (optimal: 0)

    # Write metrics to file
    with open(log_file, 'w') as f:
        f.write(f"Evaluation: {model_name} model\n\n")
        f.write(f"Mean Rank: {mean_rank:.4f}\n")
        f.write(f"MRR: {mrr:.4f}\n")
        f.write(f"Hits@10: {hits_at_10:.4f}\n")
        f.write(f"Mean Squared Error (MSE): {mse_value:.4f}")

    print(f"Evaluation metrics saved to {log_file}")

def test_model(model, mixed_df, entity2id, relation2id, model_name, log_file):
    """
    Tests the model on a dataset containing both positive and negative samples, labeled with a 'Label' column.
    """
    max_processes = os.cpu_count()

    # Separate positive and negative samples based on the 'Label' column
    positive_df = mixed_df[mixed_df['Label'] == 1]
    negative_df = mixed_df[mixed_df['Label'] == 0]

    # Test on positive samples
    print("Testing on positive samples...")
    pos_mean_rank, pos_mrr = evaluate_ranking_parallel(model, positive_df, entity2id, relation2id,
                                                       num_processes=max_processes, model_name=model_name)
    pos_hits_at_10 = hits_at_k_parallel(model, positive_df, entity2id, relation2id, k=10, num_processes=max_processes, model_name=model_name)
    pos_mse_value = mean_squared_error_parallel(model, positive_df, entity2id, relation2id, num_processes=max_processes, model_name=model_name)

    # Test on negative samples
    print("Testing on negative samples...")
    neg_mean_rank, neg_mrr = evaluate_ranking_parallel(model, negative_df, entity2id, relation2id,
                                                       num_processes=max_processes, model_name=model_name)
    neg_hits_at_10 = hits_at_k_parallel(model, negative_df, entity2id, relation2id, k=10, num_processes=max_processes, model_name=model_name)
    neg_mse_value = mean_squared_error_parallel(model, negative_df, entity2id, relation2id, num_processes=max_processes, model_name=model_name)

    # Log the metrics
    with open(log_file, 'w') as f:
        f.write(f"Test Results: {model_name} model\n\n")
        f.write(f"--- Positive Samples ---\n")
        f.write(f"Mean Rank: {pos_mean_rank:.4f}\n")
        f.write(f"MRR: {pos_mrr:.4f}\n")
        f.write(f"Hits@10: {pos_hits_at_10:.4f}\n")
        f.write(f"Mean Squared Error (MSE): {pos_mse_value:.4f}\n\n")

        f.write(f"--- Negative Samples ---\n")
        f.write(f"Mean Rank: {neg_mean_rank:.4f}\n")
        f.write(f"MRR: {neg_mrr:.4f}\n")
        f.write(f"Hits@10: {neg_hits_at_10:.4f}\n")
        f.write(f"Mean Squared Error (MSE): {neg_mse_value:.4f}\n")

    print(f"Test metrics saved to {log_file}")


# Mean Rank and Mean Reciprocal Ranking (MRR)
# The metrics evaluate how well a model ranks the correct tail entity.
# To compute these, we need to consider all possible tail entities for each (head, relation) pair and see where the correct tail is ranked.
# Note: very time-consuming for big datasets
def compute_rank_for_triple(row, model, entity2id, relation2id, model_name='TransE'):
    head = row['Subject']
    relation = row['Predicate']
    true_tail = row['Object']

    head_id = entity2id[head]
    relation_id = relation2id[relation]
    true_tail_id = entity2id[true_tail]

    if model_name == 'TransE':
        head_embed = model.entity_embeddings(torch.tensor([head_id]).long())
        relation_embed = model.relation_embeddings(torch.tensor([relation_id]).long())
        scores = []
        for tail in entity2id:
            tail_id = entity2id[tail]
            tail_embed = model.entity_embeddings(torch.tensor([tail_id]).long())
            score = torch.norm(head_embed + relation_embed - tail_embed, p=2).item()
            scores.append((tail_id, score))

    elif model_name == 'ComplEX':
        head_real = model.entity_embeddings_real(torch.tensor([head_id]).long())
        head_imag = model.entity_embeddings_imag(torch.tensor([head_id]).long())
        relation_real = model.relation_embeddings_real(torch.tensor([relation_id]).long())
        relation_imag = model.relation_embeddings_imag(torch.tensor([relation_id]).long())
        scores = []
        for tail in entity2id:
            tail_id = entity2id[tail]
            tail_real = model.entity_embeddings_real(torch.tensor([tail_id]).long())
            tail_imag = model.entity_embeddings_imag(torch.tensor([tail_id]).long())
            score_real = head_real * relation_real * tail_real + head_imag * relation_imag * tail_real
            score_imag = head_real * relation_imag * tail_imag - head_imag * relation_real * tail_imag
            score = torch.sum(score_real) + torch.sum(score_imag)
            scores.append((tail_id, score.item()))

    elif model_name == 'RotatE':
        head_embed = model.entity_embeddings(torch.tensor([head_id]).long())
        tail_embed = model.entity_embeddings(torch.tensor([true_tail_id]).long())
        phase_relation = model.phase_relation(torch.tensor([relation_id]).long())

        re_head, im_head = torch.chunk(head_embed, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail_embed, 2, dim=1)
        re_relation, im_relation = torch.cos(phase_relation), torch.sin(phase_relation)

        scores = []
        for tail in entity2id:
            tail_id = entity2id[tail]
            tail_embed = model.entity_embeddings(torch.tensor([tail_id]).long())
            re_tail, im_tail = torch.chunk(tail_embed, 2, dim=1)

            re_score = re_head * re_relation - im_head * im_relation - re_tail
            im_score = re_head * im_relation + im_head * re_relation - im_tail

            score = torch.norm(torch.stack([re_score, im_score], dim=0), dim=0).sum().item()
            scores.append((tail_id, score))

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Sort by score (ascending order)
    scores.sort(key=lambda x: x[1])

    # Get the rank of the true tail entity
    rank = [x[0] for x in scores].index(true_tail_id) + 1  # Rank is 1-based
    reciprocal_rank = 1.0 / rank

    return rank, reciprocal_rank


def evaluate_ranking_parallel(model, triples_df, entity2id, relation2id, num_processes=4, model_name='TransE'):
    # Use `partial` to fix the additional arguments for `compute_rank_for_triple`
    compute_rank_partial = partial(compute_rank_for_triple, model=model, entity2id=entity2id, relation2id=relation2id,
                                   model_name=model_name)

    with mp.Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(compute_rank_partial, [row for _, row in triples_df.iterrows()]),
                           total=len(triples_df), desc="Evaluating Ranks", ncols=100):
            results.append(result)

    # Unpack the results
    ranks, reciprocal_ranks = zip(*results)

    mean_rank = np.mean(ranks)
    mrr = np.mean(reciprocal_ranks)

    return mean_rank, mrr


# HitsAtK
# This metric measures the proportion of correct tail entities that appear in the top K predictions.
def compute_hits_for_triple(row, model, entity2id, relation2id, k, model_name='TransE'):
    head = row['Subject']
    relation = row['Predicate']
    true_tail = row['Object']

    head_id = entity2id[head]
    relation_id = relation2id[relation]
    true_tail_id = entity2id[true_tail]

    if model_name == 'TransE':
        head_embed = model.entity_embeddings(torch.tensor([head_id]).long())
        relation_embed = model.relation_embeddings(torch.tensor([relation_id]).long())
        scores = []
        for tail in entity2id:
            tail_id = entity2id[tail]
            tail_embed = model.entity_embeddings(torch.tensor([tail_id]).long())
            score = torch.norm(head_embed + relation_embed - tail_embed, p=2).item()
            scores.append((tail_id, score))

    elif model_name == 'ComplEX':
        head_real = model.entity_embeddings_real(torch.tensor([head_id]).long())
        head_imag = model.entity_embeddings_imag(torch.tensor([head_id]).long())
        relation_real = model.relation_embeddings_real(torch.tensor([relation_id]).long())
        relation_imag = model.relation_embeddings_imag(torch.tensor([relation_id]).long())
        scores = []
        for tail in entity2id:
            tail_id = entity2id[tail]
            tail_real = model.entity_embeddings_real(torch.tensor([tail_id]).long())
            tail_imag = model.entity_embeddings_imag(torch.tensor([tail_id]).long())
            score_real = head_real * relation_real * tail_real + head_imag * relation_imag * tail_real
            score_imag = head_real * relation_imag * tail_imag - head_imag * relation_real * tail_imag
            score = torch.sum(score_real) + torch.sum(score_imag)
            scores.append((tail_id, score.item()))

    elif model_name == 'RotatE':
        head_embed = model.entity_embeddings(torch.tensor([head_id]).long())
        phase_relation = model.phase_relation(torch.tensor([relation_id]).long())
        scores = []
        for tail in entity2id:
            tail_id = entity2id[tail]
            tail_embed = model.entity_embeddings(torch.tensor([tail_id]).long())
            re_head, im_head = torch.chunk(head_embed, 2, dim=1)
            re_tail, im_tail = torch.chunk(tail_embed, 2, dim=1)
            re_relation, im_relation = torch.cos(phase_relation), torch.sin(phase_relation)

            re_score = re_head * re_relation - im_head * im_relation - re_tail
            im_score = re_head * im_relation + im_head * re_relation - im_tail
            score = torch.norm(torch.stack([re_score, im_score], dim=0), dim=0).sum().item()
            scores.append((tail_id, score))

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Sort by score (ascending order)
    scores.sort(key=lambda x: x[1])

    # Check if the true tail is in the top K predictions
    top_k_tails = [x[0] for x in scores[:k]]
    return 1 if true_tail_id in top_k_tails else 0


def hits_at_k_parallel(model, triples_df, entity2id, relation2id, k, num_processes=4, model_name='TransE'):
    # Use `partial` to freeze the model, entity2id, relation2id, k, and model_name arguments
    compute_hits_partial = partial(compute_hits_for_triple, model=model, entity2id=entity2id, relation2id=relation2id,
                                   k=k, model_name=model_name)

    with mp.Pool(processes=num_processes) as pool:
        hits = sum(
            tqdm(pool.imap_unordered(compute_hits_partial, [row for _, row in triples_df.iterrows()]),
                 total=len(triples_df), desc=f"Calculating Hits@{k}", ncols=100)
        )

    hits_at_k_score = hits / len(triples_df)
    return hits_at_k_score


# MSE (Mean Squared Error)
# Although MSE is not typically use in LP, if we interpret model scores as distances and want to minimize them
# for correct triples, we can compute it for positive samples
def compute_squared_error_for_triple(row, model, entity2id, relation2id, model_name='TransE'):
    head = row['Subject']
    relation = row['Predicate']
    true_tail = row['Object']

    head_id = entity2id[head]
    relation_id = relation2id[relation]
    true_tail_id = entity2id[true_tail]

    if model_name == 'TransE':
        head_embed = model.entity_embeddings(torch.tensor([head_id]).long())
        relation_embed = model.relation_embeddings(torch.tensor([relation_id]).long())
        tail_embed = model.entity_embeddings(torch.tensor([true_tail_id]).long())
        score = torch.norm(head_embed + relation_embed - tail_embed, p=2).item()
        squared_error = score ** 2


    elif model_name == 'ComplEX':
        # Detach tensors to avoid serialization issues with autograd
        head_real = model.entity_embeddings_real(torch.tensor([head_id]).long()).detach()
        head_imag = model.entity_embeddings_imag(torch.tensor([head_id]).long()).detach()
        relation_real = model.relation_embeddings_real(torch.tensor([relation_id]).long()).detach()
        relation_imag = model.relation_embeddings_imag(torch.tensor([relation_id]).long()).detach()
        tail_real = model.entity_embeddings_real(torch.tensor([true_tail_id]).long()).detach()
        tail_imag = model.entity_embeddings_imag(torch.tensor([true_tail_id]).long()).detach()

        score_real = head_real * relation_real * tail_real + head_imag * relation_imag * tail_real
        score_imag = head_real * relation_imag * tail_imag - head_imag * relation_real * tail_imag

        score = torch.norm(score_real, p=2) + torch.norm(score_imag, p=2)
        squared_error = score.item() ** 2

    elif model_name == 'RotatE':
        head_embed = model.entity_embeddings(torch.tensor([head_id]).long())
        tail_embed = model.entity_embeddings(torch.tensor([true_tail_id]).long())
        phase_relation = model.phase_relation(torch.tensor([relation_id]).long())

        re_head, im_head = torch.chunk(head_embed, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail_embed, 2, dim=1)
        re_relation, im_relation = torch.cos(phase_relation), torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation - re_tail
        im_score = re_head * im_relation + im_head * re_relation - im_tail

        score = torch.norm(torch.stack([re_score, im_score], dim=0), dim=0).sum().item()
        squared_error = score ** 2

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return squared_error


def mean_squared_error_parallel(model, triples_df, entity2id, relation2id, num_processes=4, model_name='TransE'):
    # Use `partial` to freeze the model, entity2id, relation2id, and model_name arguments
    compute_squared_error_partial = partial(compute_squared_error_for_triple, model=model, entity2id=entity2id,
                                            relation2id=relation2id, model_name=model_name)

    with mp.Pool(processes=num_processes) as pool:
        squared_errors = list(
            tqdm(pool.imap_unordered(compute_squared_error_partial, [row for _, row in triples_df.iterrows()]),
                 total=len(triples_df), desc="Calculating MSE", ncols=100)
        )

    mse = sum(squared_errors) / len(triples_df)
    return mse

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
    model_name = 'TransE'  # You can choose between 'TransE' (default), 'ComplEX' or 'RotatE'

    # Define the directory for saving metrics
    metrics_dir = "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Create entity embeddings using the specified model
    model, entity2id, relation2id = create_entity_embeddings(triples_df_train, model_name=model_name)

    # Train the model
    train_model_batch(
        model,
        triples_df_train,
        entity2id,
        relation2id,
        model_name=model_name,
        epochs=100,
        learning_rate=0.001,
        margin=1.0,
        batch_size=128,
        log_file=os.path.join(metrics_dir, f"{model_name}_training_log.txt")
    )

    # Optionally, save the model and embeddings
    save_model_and_embeddings(model, model_name=model_name)

    # Combine positive and negative validation samples into a single mixed dataframe with labels
    validation_df_mixed = combine_positive_negative_samples(triples_df_valid, triples_df_valid_negatives)

    # Evaluate the link prediction model and save metrics to a file
    evaluate_model(
        model,
        validation_df_mixed,
        entity2id,
        relation2id,
        model_name=model_name,
        log_file=os.path.join(metrics_dir, f"{model_name}_evaluation_log.txt")
    )

    # Combine positive and negative test samples into a single mixed dataframe with labels
    test_df_mixed = combine_positive_negative_samples(triples_df_test, triples_df_test_negatives)

    # Test the link prediction model and save the metrics to a file
    test_model(
        model,
        test_df_mixed,
        entity2id,
        relation2id,
        model_name=model_name,
        log_file=os.path.join(metrics_dir, f"{model_name}_test_log.txt")
    )

    # Optionally, predict the tail of a specific triple
    #subject = 'Entity1'
    #predicate = 'Predicate1'
    #predicted_tail = predict_tail(model, subject, predicate, entity2id, relation2id, model_name=model_name)
    #print(f"Predicted tail entity: {predicted_tail}")

    # Optionally, load the model and embeddings and re-evaluate the loaded model
    #loaded_model = load_model_and_embeddings(len(entity2id), len(relation2id), embedding_dim=100, model_name=model_name)
    #evaluate_model(loaded_model, validation_df_mixed, entity2id, relation2id, model_name=model_name, log_file=os.path.join(metrics_dir, f"loaded_{model_name}_evaluation_log.txt"))
    #test_model(loaded_model,test_df_mixed,entity2id,relation2id,model_name=model_name,log_file=os.path.join(metrics_dir, f"loaded_{model_name}_test_log.txt"))

    # Optionally, predict the tail of a specific triple using the loaded model
    #predicted_tail_loaded = predict_tail(loaded_model, subject, predicate, entity2id, relation2id)
    #print(f"Predicted tail entity with loaded model: {predicted_tail_loaded}")