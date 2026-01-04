import os
import json
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_net = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        # x: [batch, seq_len, d_model]
        # mask: [batch, seq_len]
        attn_logits = self.attention_net(x).squeeze(2)  # [batch, seq_len]
        
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
        # 1. CLS Branch
        z_cls = self.cls_projector(cls_embedding)
        
        # 2. Token Branch
        #  input 为 [batch, channels, length]
        tok_emb_permuted = token_embeddings.permute(0, 2, 1)
        refined_tok_emb = self.token_refiner(tok_emb_permuted).permute(0, 2, 1)
        z_tok_pooled = self.attention_pooling(refined_tok_emb, mask)
        z_tok = self.tok_projector(z_tok_pooled)
        
        # 3. Fusion
        z_fused_concat = torch.cat([z_cls, z_tok], dim=1)
        gate_values = self.gate(z_fused_concat)
        z_fused_gated = z_fused_concat * gate_values
        
        # 4. Classification
        return self.classifier_head(z_fused_gated)

class ProteinPredictor:
    def __init__(self, model_checkpoint_path, label_map_path, esm_model_name, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Setting up predictor on {self.device} ---")

        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        num_classes = len(self.label_map)

        print(f"Loading ESM model: {esm_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        self.esm_model = AutoModel.from_pretrained(esm_model_name).to(self.device)
        self.esm_model.eval()

        esm_config = AutoConfig.from_pretrained(esm_model_name)
        d_model = esm_config.hidden_size

        print(f"Loading classifier weights from: {model_checkpoint_path}")
        self.classifier = ProtDualBranchEnhancedClassifier(
            d_model=d_model,
            projection_dim=32,    # Default from Config
            num_classes=num_classes,
            dropout=0.0,    
            kernel_size=3      
        ).to(self.device)

        state_dict = torch.load(model_checkpoint_path, map_location=self.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()
        print("--- Model loaded successfully ---")

    def predict_sequence(self, sequence, max_len=1024):
        
        inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
        
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            # [Batch=1, Seq_Len, Hidden]
            cls_embedding = hidden_states[:, 0, :]
            token_embeddings = hidden_states[:, 1:-1, :]
            
            seq_len = token_embeddings.shape[1]
            mask = torch.ones((1, seq_len), dtype=torch.long).to(self.device)

            logits = self.classifier(cls_embedding, token_embeddings, mask)
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
            pred_label = self.idx_to_label[pred_idx.item()]
            
            return {
                "label": pred_label,
                "confidence": confidence.item(),
                "probabilities": probs.cpu().numpy().tolist()[0]
            }

def main():
    parser = argparse.ArgumentParser(description="Inference script for Protein Classification.")
    parser.add_argument("--input_fasta", type=str, required=True, help="Path to input FASTA file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument("--label_map", type=str, required=True, help="Path to label_map.json.")
    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D", help="HuggingFace model name used for training (e.g. facebook/esm2_t33_650M_UR50D).")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Where to save results.")
    parser.add_argument("--max_len", type=int, default=1024, help="Max sequence length.")
    
    args = parser.parse_args()

    predictor = ProteinPredictor(
        model_checkpoint_path=args.checkpoint,
        label_map_path=args.label_map,
        esm_model_name=args.esm_model
    )

    results = []
    
    print(f"\nProcessing sequences from {args.input_fasta}...")
    fasta_sequences = list(SeqIO.parse(open(args.input_fasta), 'fasta'))
    
    for record in tqdm(fasta_sequences, ncols=100):
        seq = str(record.seq)
        
        if "prot_" in args.esm_model.lower():
            seq = " ".join(list(seq))
            
        try:
            prediction = predictor.predict_sequence(seq, max_len=args.max_len)
            
            results.append({
                "Protein_ID": record.id,
                "Predicted_Label": prediction['label'],
                "Confidence": f"{prediction['confidence']:.4f}",
                "Sequence_Length": len(record.seq)
            })
        except Exception as e:
            print(f"Error processing {record.id}: {e}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nDone! Predictions saved to {args.output_csv}")
    print(df.head())

if __name__ == "__main__":
    main()
