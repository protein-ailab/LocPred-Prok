import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report,
                             matthews_corrcoef)
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig

# --- 1. Configuration ---
class Config:
    """Stores fixed hyperparameters related to the training process."""
    ARCHITECTURE = 'dual_branch_enhanced'

    # --- Model Architecture Parameters ---
    DROPOUT = 0.3
    PROJECTION_DIM = 32
    CONV_KERNEL_SIZE = 3

    # --- Training Process Parameters ---
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    VAL_SIZE_FROM_TRAIN = 0.1
    PATIENCE = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42

    # --- Target groups for subgroup analysis (keyword -> display name) ---
    SUBGROUP_MAP = {
        'archaea': 'Archaea',
        'positive': 'Gram positive',
        'negative': 'Gram negative'
    }

# --- 2. Model Definitions ---
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_net = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        attn_logits = self.attention_net(x).squeeze(2)
        attn_logits.masked_fill_(mask == 0, -float('inf'))
        attn_weights = F.softmax(attn_logits, dim=1)
        return torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

class ProtDualBranchEnhancedClassifier(nn.Module):
    def __init__(self, d_model, projection_dim, num_classes, dropout, kernel_size):
        super().__init__()
        self.cls_projector = nn.Linear(d_model, projection_dim)
        self.token_refiner = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding='same'),
            nn.ReLU()
        )
        self.attention_pooling = AttentionPooling(d_model)
        self.tok_projector = nn.Linear(d_model, projection_dim)
        fused_dim = projection_dim * 2
        self.gate = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.Sigmoid())
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, num_classes)
        )

    def forward(self, cls_embedding, token_embeddings, mask):
        z_cls = self.cls_projector(cls_embedding)
        tok_emb_permuted = token_embeddings.permute(0, 2, 1)
        refined_tok_emb = self.token_refiner(tok_emb_permuted).permute(0, 2, 1)
        z_tok_pooled = self.attention_pooling(refined_tok_emb, mask)
        z_tok = self.tok_projector(z_tok_pooled)
        z_fused_concat = torch.cat([z_cls, z_tok], dim=1)
        gate_values = self.gate(z_fused_concat)
        z_fused_gated = z_fused_concat * gate_values
        return self.classifier_head(z_fused_gated)

# --- 3. Data Handling ---
class DynamicProteinDataset(Dataset):
    def __init__(self, metadata, protein_ids):
        self.metadata = metadata
        self.protein_ids = protein_ids

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        info = self.metadata[protein_id]
        cls_embedding = torch.load(info['cls_emb_path'])
        token_embedding = torch.load(info['tok_emb_path'])
        label = info['label_int']
        return cls_embedding, token_embedding, label

def protein_collate_fn(batch):
    cls_embeddings = [item[0] for item in batch]
    token_embeddings = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    cls_embeddings_batch = torch.stack(cls_embeddings, dim=0)
    padded_token_embeddings = pad_sequence(token_embeddings, batch_first=True, padding_value=0.0)
    lengths = [len(seq) for seq in token_embeddings]
    mask = torch.zeros(padded_token_embeddings.shape[0], padded_token_embeddings.shape[1], dtype=torch.long)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    labels = torch.tensor(labels, dtype=torch.long)
    return cls_embeddings_batch, padded_token_embeddings, mask, labels

# --- 4. Training and Evaluation Framework ---
def calculate_per_class_mcc(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_mcc = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        per_class_mcc[i] = 0.0 if denominator == 0 else numerator / denominator
    return per_class_mcc

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False, ncols=100)
    for batch in progress_bar:
        cls_emb, tok_emb, masks, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        logits = model(cls_emb, tok_emb, masks)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)
        progress_bar.set_postfix(loss=loss.item(), acc=total_correct/total_samples)
    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in dataloader:
            cls_emb, tok_emb, masks, labels = [b.to(device) for b in batch]
            logits = model(cls_emb, tok_emb, masks)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)[2]
    f1_macro = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[2]
    mcc = matthews_corrcoef(all_labels, all_preds)
    per_class_mcc = calculate_per_class_mcc(all_labels, all_preds, num_classes)

    metrics = {
        "loss": avg_loss, "accuracy": accuracy, "f1_weighted": f1_weighted,
        "f1_macro": f1_macro, "mcc": mcc, "per_class_mcc": per_class_mcc
    }
    return metrics, all_labels, all_preds

# --- 5. Main Execution Function ---
def run_experiment(args, model_name):
    config = Config()
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # --- Automatically retrieve embedding dimension ---
    print("--- Automatically retrieving embedding dimension (d_model) ---")
    try:
        model_config = AutoConfig.from_pretrained(model_name)
        d_model = model_config.hidden_size
        print(f"Successfully retrieved! d_model for model '{model_name}' is: {d_model}")
    except Exception as e:
        print(f"Failed to automatically retrieve d_model: {e}")
        return

    print("\n" + "="*60)
    print(f"Starting experiment for: {model_name}")
    print("="*60)

    # --- Dynamically construct paths ---
    model_short_name = model_name.split('/')[-1]
    data_dir = os.path.join(args.base_feature_dir, model_short_name)
    train_metadata_path = os.path.join(data_dir, 'full_dataset_metadata.json')
    test_metadata_path = os.path.join(data_dir, 'benchmarking_dataset_metadata.json')
    label_map_path = os.path.join(args.base_feature_dir, 'label_map.json')

    print("--- 1. Loading Data ---")
    with open(train_metadata_path, 'r') as f:
        train_metadata = json.load(f)
    with open(test_metadata_path, 'r') as f:
        test_metadata = json.load(f)
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    print("\n--- 2. Initializing Model ---")
    model = ProtDualBranchEnhancedClassifier(
        d_model=d_model,
        projection_dim=config.PROJECTION_DIM,
        num_classes=num_classes,
        dropout=config.DROPOUT,
        kernel_size=config.CONV_KERNEL_SIZE
    ).to(config.DEVICE)

    print(f"Using PLM: {model_name}")
    print(f"Downstream Classifier: {config.ARCHITECTURE}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_model_path = f"best_model_{model_short_name}.pth"

    if args.mode == 'train':
        print("\n--- Mode: Training ---")
        train_val_ids = list(train_metadata.keys())
        train_ids, val_ids = train_test_split(train_val_ids, test_size=config.VAL_SIZE_FROM_TRAIN, random_state=config.RANDOM_SEED)
        train_dataset = DynamicProteinDataset(train_metadata, train_ids)
        val_dataset = DynamicProteinDataset(train_metadata, val_ids)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=protein_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=protein_collate_fn)
        print(f"Training set: {len(train_ids)} | Validation set: {len(val_ids)}")

        print("\n--- 2a. Calculating Class Weights ---")
        class_counts = np.bincount([train_metadata[pid]['label_int'] for pid in train_ids], minlength=num_classes)
        class_weights = torch.tensor([1.0 / (c if c > 0 else 1) for c in class_counts], dtype=torch.float).to(config.DEVICE)
        class_weights /= class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Calculated class weights have been applied to the loss function.")

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5)

        print("\n--- 4. Starting Training ---")
        best_val_f1_macro = -1
        epochs_no_improve = 0
        for epoch in range(config.EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
            val_metrics, _, _ = evaluate(model, val_loader, criterion, config.DEVICE, num_classes)
            print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val F1-Macro: {val_metrics['f1_macro']:.4f}")
            scheduler.step(val_metrics['f1_macro'])
            if val_metrics['f1_macro'] > best_val_f1_macro:
                best_val_f1_macro = val_metrics['f1_macro']
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Validation F1-Macro improved, model saved to {best_model_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= config.PATIENCE:
                print(f"Validation performance did not improve for {config.PATIENCE} consecutive epochs, triggering early stopping.")
                break

    elif args.mode == 'test':
        print("\n--- Mode: Test Only ---")
        if not os.path.exists(best_model_path):
            print(f"Warning: Model file {best_model_path} not found. Skipping test for {model_short_name}.")
            return
        print(f"Loading model weights from '{best_model_path}'.")

    print("\n--- 5. Final Evaluation on Test Set ---")
    test_ids = list(test_metadata.keys())
    test_dataset = DynamicProteinDataset(test_metadata, test_ids)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=protein_collate_fn)
    print(f"Number of test samples: {len(test_ids)}")

    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, config.DEVICE, num_classes)

    print("\n" + "="*50)
    print(f"--- {model_short_name} Overall Test Set Performance Report ---")
    print("="*50)
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  - Weighted F1: {test_metrics['f1_weighted']:.4f}")
    print(f"  - Macro F1: {test_metrics['f1_macro']:.4f}")
    print(f"  - Matthews Correlation Coefficient (MCC): {test_metrics['mcc']:.4f}")
    print("\nPer-Class MCC:")
    for class_idx, mcc_score in test_metrics['per_class_mcc'].items():
        print(f"  - MCC for class '{idx_to_label.get(class_idx)}': {mcc_score:.4f}")
    target_names = [idx_to_label.get(i) for i in range(num_classes)]
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print("\nClassification Report:")
    print(report)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

    print("\n" + "="*50)
    print("--- Subgroup Performance Analysis ---")
    print("="*50)
    test_results_map = {pid: (yt, yp) for pid, yt, yp in zip(test_ids, y_true, y_pred)}

    for keyword, display_name in config.SUBGROUP_MAP.items():
        subgroup_y_true, subgroup_y_pred = [], []
        for pid in test_ids:
            organism_type_from_meta = test_metadata[pid].get('organism_type', '').lower()
            if keyword in organism_type_from_meta:
                yt, yp = test_results_map[pid]
                subgroup_y_true.append(yt)
                subgroup_y_pred.append(yp)

        print(f"\n--- Analyzing Subgroup: '{display_name}' (n = {len(subgroup_y_true)}) ---")
        if not subgroup_y_true:
            print("No samples for this subgroup found in the test set.")
            continue

        sub_accuracy = accuracy_score(subgroup_y_true, subgroup_y_pred)
        sub_f1_macro = precision_recall_fscore_support(subgroup_y_true, subgroup_y_pred, average='macro', zero_division=0)[2]
        sub_mcc = matthews_corrcoef(subgroup_y_true, subgroup_y_pred)

        print(f"  - Accuracy: {sub_accuracy:.4f}")
        print(f"  - Macro F1: {sub_f1_macro:.4f}")
        print(f"  - Matthews Correlation Coefficient (MCC): {sub_mcc:.4f}")

        print("\n  Sample counts per class in this subgroup:")
        sub_class_counts = np.bincount(subgroup_y_true, minlength=num_classes)
        for class_idx, count in enumerate(sub_class_counts):
            class_name = idx_to_label.get(class_idx, f"Unknown Class {class_idx}")
            print(f"    - {class_name}: {count}")

        sub_per_class_mcc = calculate_per_class_mcc(subgroup_y_true, subgroup_y_pred, num_classes)

        print("\n  MCC per class:")
        for class_idx, mcc_score in sub_per_class_mcc.items():
            class_name = idx_to_label.get(class_idx, f"Unknown Class {class_idx}")
            print(f"    - {class_name}: {mcc_score:.4f}")

if __name__ == '__main__':
    # --- List of models to run ---
    model_list = [
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
    ]

    parser = argparse.ArgumentParser(description="Training and evaluation script for protein classification models (supports multiple PLMs).")
    parser.add_argument(
        '--base_feature_dir',
        type=str,
        default='./precomputed_features',
        help="Base directory where features for all models are stored."
    )
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    args = parser.parse_args()

    # Loop through the model list and run a full experiment for each model
    for model_name in model_list:
        # Create an independent copy of arguments for each experiment
        experiment_args = argparse.Namespace(**vars(args))
        experiment_args.model_name = model_name
        experiment_args.d_model = None # Force auto-detection every time
        experiment_args.model_path = None # Use auto-generated path

        run_experiment(experiment_args, model_name)
