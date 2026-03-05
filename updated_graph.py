"""
research_graph_biometrics.py

A unified and research-ready script for graph-based biometric identification.
This version integrates a corrected TripletDataset and supports both 
classification and triplet loss training modes.
"""
import torch.nn as nn
import os
import sys
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import time
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, TripletMarginLoss
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, Dataset, DataLoader, Batch
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock implementations for standalone execution
from config_and_utils import Config, set_reproducibility, create_consistent_splits, BiometricEvaluator, EarlyStopping, log_experiment_setup

class GraphDataset(Dataset):
    """Memory-efficient graph dataset that creates graphs on-the-fly."""
    
    def __init__(self, data_df: pd.DataFrame, user_id_map: Dict, 
             spacy_model, config, cache_graphs: bool = False):
        super().__init__()
        self.data = data_df.reset_index(drop=True)
        self.user_id_map = user_id_map
        self.spacy_model = spacy_model
        self.cache_graphs = cache_graphs
        self.config = config
        self._cache = {} if cache_graphs else None
        
        self.valid_indices = []
        logger.info("Pre-filtering valid graphs...")
        for idx in range(len(self.data)):
            graph = self._create_graph_from_text_safe(idx)
            if graph is not None:
                self.valid_indices.append(idx)
                if self.cache_graphs:
                    self._cache[idx] = graph
        
        logger.info(f"Found {len(self.valid_indices)} valid graphs out of {len(self.data)}")
    
    def len(self):
        return len(self.valid_indices)
    
    def get(self, idx):
        actual_idx = self.valid_indices[idx]
        if self.cache_graphs and actual_idx in self._cache:
            return self._cache[actual_idx]
        return self._create_graph_from_text_safe(actual_idx)
    
    def _create_graph_from_text_safe(self, actual_idx):
        try:
            row = self.data.iloc[actual_idx]
            return self._create_graph_from_text(row['prompt'], self.user_id_map, row['user_id'], self.config)
        except Exception as e:
            logger.debug(f"Failed to create graph for index {actual_idx}: {e}")
            return None
    
    def _create_graph_from_text(self, text: str, user_id_map: Dict, user_id: str, config) -> Optional[Data]:
        try:
            doc = self.spacy_model(str(text))
            node_features, token_to_idx = [], {}
            idx = 0
            
            for token in doc:
                if token.has_vector and not token.is_space:
                    word_vec = token.vector
                    
                    if config.ablate_pos:
                       pos_feat = np.zeros(4)
                    else:
                       pos_feat = [1.0 if token.pos_ == pos else 0.0 for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']]

                    if config.ablate_dep:
                       dep_feat = np.zeros(4)
                    else:
                      dep_feat = [1.0 if token.dep_ == dep else 0.0 for dep in ['ROOT', 'nsubj', 'dobj', 'amod']]
                    features = np.concatenate([word_vec, pos_feat, dep_feat, [float(len(token.text)), float(token.is_stop), float(token.is_alpha)]])
                    node_features.append(features)
                    token_to_idx[token] = idx
                    idx += 1
            
            if len(node_features) < 2: return None
            
            edge_sources, edge_targets, edge_types = [], [], []
            edge_type_map = {'ROOT': 0, 'nsubj': 1, 'dobj': 2, 'amod': 3, 'other': 4}
            
            for token in doc:
                if token in token_to_idx:
                    for child in token.children:
                        if child in token_to_idx:
                            edge_sources.extend([token_to_idx[token], token_to_idx[child]])
                            edge_targets.extend([token_to_idx[child], token_to_idx[token]])
                            edge_type = edge_type_map.get(child.dep_, 4)
                            edge_types.extend([edge_type, edge_type])
                    
                    if token.head != token and token.head in token_to_idx:
                        edge_sources.extend([token_to_idx[token], token_to_idx[token.head]])
                        edge_targets.extend([token_to_idx[token.head], token_to_idx[token]])
                        edge_type = edge_type_map.get(token.dep_, 4)
                        edge_types.extend([edge_type, edge_type])

            x = torch.tensor(np.array(node_features), dtype=torch.float32)
            
            if edge_sources:
                edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
            else: # Self-loops if no edges
                num_nodes = len(node_features)
                edge_index = torch.arange(0, num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)

            y_val = user_id_map.get(user_id, -1)
            if y_val == -1: return None
            y = torch.tensor([y_val], dtype=torch.long)

            return Data(x=x, edge_index=edge_index, y=y)
            
        except Exception as e:
            logger.debug(f"Error creating graph: {e}")
            return None

### --- TRIPLET DATASET AND COLLATE FUNCTIONS --- ###

class TripletDataset(TorchDataset):
    """
    Fixed PyTorch Dataset for triplets of (anchor, positive, negative) graphs.
    Ensures anchor is never selected as positive.
    """
    def __init__(self, graph_dataset, seed=42):
        self.graph_dataset = graph_dataset
        self.rng = np.random.RandomState(seed)
        
        # This dataset requires pre-computed mappings. Ensure they exist.
        if not all(hasattr(graph_dataset, attr) for attr in ['labels', 'labels_to_indices', 'unique_labels']):
            raise AttributeError("The provided graph_dataset must be pre-processed with .labels, .labels_to_indices, and .unique_labels attributes.")
            
        self.valid_triplets = []
        for idx in range(len(graph_dataset)):
            anchor_label = self.graph_dataset.labels[idx]
            if anchor_label == -1: continue

            positive_candidates = [i for i in self.graph_dataset.labels_to_indices[anchor_label] if i != idx]
            if not positive_candidates: continue

            negative_labels = [l for l in self.graph_dataset.unique_labels if l != anchor_label and l != -1]
            if not negative_labels: continue
            
            self.valid_triplets.append({
                'anchor_idx': idx,
                'positive_candidates': positive_candidates,
                'negative_labels': negative_labels
            })
        
        logger.info(f"Created {len(self.valid_triplets)} valid triplets from {len(graph_dataset)} graphs")

    def __len__(self):
        return len(self.valid_triplets)

    def __getitem__(self, index):
        triplet_info = self.valid_triplets[index]
        
        anchor_idx = triplet_info['anchor_idx']
        anchor_graph = self.graph_dataset.get(anchor_idx)
        
        positive_idx = self.rng.choice(triplet_info['positive_candidates'])
        positive_graph = self.graph_dataset.get(positive_idx)
        
        negative_label = self.rng.choice(triplet_info['negative_labels'])
        negative_indices = self.graph_dataset.labels_to_indices[negative_label]
        negative_idx = self.rng.choice(negative_indices)
        negative_graph = self.graph_dataset.get(negative_idx)
        
        # Essential validation
        assert anchor_idx != positive_idx, "Anchor and positive are the same!"
        assert anchor_graph.y.item() == positive_graph.y.item(), "Positive has wrong label!"
        assert anchor_graph.y.item() != negative_graph.y.item(), "Negative has same label!"
        
        return anchor_graph, positive_graph, negative_graph

def custom_collate(batch):
    """Custom collate function that filters out None values for classification."""
    batch = [item for item in batch if item is not None]
    return Batch.from_data_list(batch) if batch else None

def custom_triplet_collate(batch):
    """Custom collate function for triplet batches."""
    anchors, positives, negatives = zip(*batch)
    anchor_batch = Batch.from_data_list(anchors)
    positive_batch = Batch.from_data_list(positives)
    negative_batch = Batch.from_data_list(negatives)
    return anchor_batch, positive_batch, negative_batch

### --- GNN MODEL AND MAIN BIOMETRICS CLASS --- ###

class ImprovedGraphBiometricModel(torch.nn.Module):
    """Enhanced Graph Neural Network for biometric identification."""
    def __init__(self, node_feature_dim: int, hidden_channels: int, 
                 embedding_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.node_projection = Linear(node_feature_dim, hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels // 2, heads=4, concat=True)
        self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels)
        
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.embedding_head = Linear(hidden_channels * 3, embedding_dim)
        self.classifier_head = Linear(embedding_dim, num_classes)
        self.dropout = dropout
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: torch.nn.init.zeros_(m.bias)
    
    def forward(self, data):
        """Generates both classification logits and style embeddings."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.node_projection(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        identity = x
        x = self.conv1(x, edge_index)
        if x.size(0) > 1: x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        
        x = self.conv2(x, edge_index)
        if x.size(0) > 1: x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        if x.size(0) > 1: x = self.bn3(x)
        x = F.relu(x)
        
        graph_rep = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
        
        style_embedding = F.relu(self.embedding_head(graph_rep))
        logits = self.classifier_head(F.dropout(style_embedding, p=self.dropout, training=self.training))
        
        return logits, style_embedding

class GraphBiometrics(nn.Module):
    def __init__(self, config):
        super(GraphBiometrics, self).__init__()
        self.config = config
        self.device = self.config.get_device()
        self.pooling_strategy = config.pooling
        self.spacy_model = None  # <== Add this
        self._setup_spacy()
        in_channels = config.node_feature_dim   # e.g., 311 (spaCy vector + POS one-hot)
        hidden_channels = config.GRAPH['gnn_hidden_channels']
        num_classes = config.num_users

        # GNN layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Pooling strategy
        if self.pooling_strategy == "attention":
            self.att_gate = nn.Linear(hidden_channels, 1)
            self.pool = GlobalAttention(gate_nn=self.att_gate)
        elif self.pooling_strategy == "max":
            self.pool = global_max_pool
        else:
            self.pool = global_mean_pool

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.pool(x, batch)
        return self.classifier(x)
    
    def _setup_spacy(self):
        model_name = "en_core_web_md"
        try:
            self.spacy_model = spacy.load(model_name)
            base_dim = self.spacy_model.vocab.vectors.shape[1]
            self.node_feature_dim = base_dim + 4 + 4 + 3 # word_vec + pos + dep + other
            logger.info(f"Loaded spaCy model '{model_name}' with {self.node_feature_dim}D node features")
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found. Please run: python -m spacy download {model_name}")
            sys.exit(1)
    
    def create_datasets(self, data_splits: Dict) -> Dict:
        datasets = {}
        for split_name in ['train', 'validation', 'test_seen', 'unseen']:
            if split_name not in data_splits or data_splits[split_name].empty:
                logger.warning(f"Split {split_name} not found or empty.")
                continue
            
            logger.info(f"Creating graph dataset for {split_name} split...")
            datasets[split_name] = GraphDataset(
                data_splits[split_name], data_splits['user_id_map'], self.spacy_model,
                self.config,  # ✅ Pass the config
                cache_graphs=(split_name == 'train')

            )
        return datasets

    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None) -> float:
        start_time = time.time()
        
        if not train_dataset or len(train_dataset) == 0:
            logger.error("No valid training graphs available")
            return 0.0

        # --- Model and Optimizer Setup ---
        num_classes = len(train_dataset.user_id_map)
        self.model = ImprovedGraphBiometricModel(
            node_feature_dim=self.node_feature_dim,
            hidden_channels=self.config.GRAPH['gnn_hidden_channels'],
            embedding_dim=self.config.GRAPH['embedding_dim'],
            num_classes=num_classes,
            dropout=self.config.GRAPH['dropout']
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.GRAPH['lr'], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.GRAPH['epochs'])
        
        # --- Training Mode Logic ---
        training_mode = self.config.GRAPH.get('training_mode', 'classification')
        logger.info(f"Starting training in '{training_mode}' mode.")

        if training_mode == 'triplet':
            # Prepare dataset with necessary attributes for TripletDataset
            logger.info("Preparing dataset for triplet training...")
            all_labels = [train_dataset.get(i).y.item() for i in range(len(train_dataset))]
            train_dataset.labels = all_labels
            train_dataset.labels_to_indices = {l: [i for i, lbl in enumerate(all_labels) if lbl == l] for l in set(all_labels)}
            train_dataset.unique_labels = list(train_dataset.labels_to_indices.keys())

            triplet_dataset = TripletDataset(train_dataset, seed=self.config.RANDOM_SEED)
            batch_size = max(1, self.config.GRAPH['batch_size'] // 3)
            train_loader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_triplet_collate, num_workers=os.cpu_count())
            criterion = TripletMarginLoss(margin=self.config.GRAPH.get('triplet_margin', 0.5))

        else: # Default to classification
            train_loader = DataLoader(train_dataset, batch_size=self.config.GRAPH['batch_size'], shuffle=True, collate_fn=custom_collate, num_workers=os.cpu_count())
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        val_loader = DataLoader(val_dataset, batch_size=self.config.GRAPH['batch_size'], shuffle=False, collate_fn=custom_collate, num_workers=os.cpu_count()) if val_dataset else None
        early_stopping = EarlyStopping(patience=7, min_delta=0.001)

        # --- Training Loop ---
        for epoch in range(1, self.config.GRAPH['epochs'] + 1):
            self.model.train()
            total_loss, total_correct, total_samples = 0, 0, 0

            for batch_data in train_loader:
                if batch_data is None: continue
                optimizer.zero_grad()
                
                if training_mode == 'triplet':
                    anchor, positive, negative = batch_data
                    anchor_emb = self.model(anchor.to(self.device))[1]
                    positive_emb = self.model(positive.to(self.device))[1]
                    negative_emb = self.model(negative.to(self.device))[1]
                    loss = criterion(anchor_emb, positive_emb, negative_emb)
                else: # Classification
                    batch_data = batch_data.to(self.device)
                    logits, _ = self.model(batch_data)
                    loss = criterion(logits, batch_data.y)
                    _, predicted = torch.max(logits.data, 1)
                    total_correct += (predicted == batch_data.y).sum().item()
                    total_samples += batch_data.y.size(0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_acc = total_correct / total_samples if total_samples > 0 else 0
            
            # --- Validation ---
            if val_loader:
                val_acc = self._validate(val_loader)
                logger.info(f'Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
                if early_stopping(val_acc, self.model):
                    logger.info(f"Early stopping at epoch {epoch}. Best validation accuracy: {early_stopping.best_score:.4f}")
                    break
            else:
                logger.info(f'Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}')

            scheduler.step()

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return training_time
    
    def _validate(self, val_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                batch = batch.to(self.device)
                logits, _ = self.model(batch)
                _, predicted = torch.max(logits.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
        return correct / total if total > 0 else 0

    def generate_embeddings(self, dataset: Dataset, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        if not dataset or len(dataset) == 0: return np.array([]), np.array([])
        
        self.model.eval()
        embeddings, labels = [], []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), collate_fn=custom_collate)
        
        with torch.no_grad():
            for batch in loader:
                if batch is None: continue
                batch = batch.to(self.device)
                _, embedding = self.model(batch)
                embeddings.append(embedding.cpu().numpy())
                labels.append(batch.y.cpu().numpy())
        
        return np.vstack(embeddings), np.concatenate(labels) if labels else np.array([])

def main():
    """Main function for standalone execution."""
    config = Config()
    # Add training_mode to config, e.g., config.GRAPH['training_mode'] = 'triplet'
    config.GRAPH['training_mode'] = 'classification' # or 'triplet'
    
    set_reproducibility(config.RANDOM_SEED)
    
    logger.info(f"Using device: {config.get_device()}")
    df = pd.read_csv(config.DATASET_PATH)
    data_splits = create_consistent_splits(df, config)
    log_experiment_setup(config, data_splits)
    
    graph_biometrics = GraphBiometrics(config)
    datasets = graph_biometrics.create_datasets(data_splits)
    
    if 'train' not in datasets or len(datasets['train']) == 0:
        logger.error("No valid training graphs created. Exiting.")
        return
    
    training_time = graph_biometrics.train(datasets['train'], datasets.get('validation'))
    
    logger.info("\nGenerating embeddings for evaluation...")
    train_embeddings, train_labels = graph_biometrics.generate_embeddings(datasets['train'])
    test_embeddings, test_labels = graph_biometrics.generate_embeddings(datasets['test_seen'])
    unseen_embeddings, _ = graph_biometrics.generate_embeddings(datasets.get('unseen'))
    
    if len(train_embeddings) > 0 and len(test_embeddings) > 0:
        evaluator = BiometricEvaluator()
        evaluator.evaluate_method(
            train_embeddings, train_labels,
            test_embeddings, test_labels,
            unseen_embeddings, "Graph-based Method", training_time
        )
        evaluator.generate_report()
    else:
        logger.error("Could not generate enough embeddings for evaluation.")

if __name__ == "__main__":
    main()