import os
import torch
import json
import argparse
from tqdm import tqdm
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel


def process_fasta(input_fasta, base_output_dir, model_name, max_len, label_map_path=None):
    """
    Processes a single FASTA file to extract and save protein features.
    Features are saved in a subdirectory named after the model.
    """
    print(f"--- Started processing file: {input_fasta} ---")
    print(f"--- Using model: {model_name} ---")

    # 1. Create a separate subdirectory based on the model name
    model_short_name = model_name.split('/')[-1]
    output_dir = os.path.join(base_output_dir, model_short_name)
    cls_emb_dir = os.path.join(output_dir, 'cls_embeddings')
    tok_emb_dir = os.path.join(output_dir, 'token_embeddings')
    os.makedirs(cls_emb_dir, exist_ok=True)
    os.makedirs(tok_emb_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # 3. Load or create the label map
    if label_map_path and os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        new_map_created = False
    else:
        label_map = {}
        new_map_created = True
    label_counter = len(label_map)

    # 4. Iterate through FASTA sequences
    metadata = {}
    fasta_sequences = SeqIO.parse(open(input_fasta), 'fasta')
    for record in tqdm(fasta_sequences, desc=f"Processing {os.path.basename(input_fasta)}", ncols=100):
        try:
            header_parts = record.description.split('|')
            if len(header_parts) < 3:
                continue
            protein_id, label_str, organism_type = header_parts[0], header_parts[1], header_parts[2]
            sequence = str(record.seq)

            # Models like ProtBERT / ProtT5 require spaces between amino acids
            if "prot_" in model_name.lower():
                sequence = " ".join(list(sequence))

            if label_str not in label_map:
                if new_map_created:
                    label_map[label_str] = label_counter
                    label_counter += 1
                else:
                    # If using a pre-existing label map, skip proteins with unknown labels
                    continue
            label_int = label_map[label_str]

            inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.last_hidden_state
            cls_embedding = hidden_states[:, 0, :]
            token_embeddings = hidden_states[:, 1:-1, :]

            cls_emb_path = os.path.join(cls_emb_dir, f"{protein_id}.pt")
            tok_emb_path = os.path.join(tok_emb_dir, f"{protein_id}.pt")
            torch.save(cls_embedding.squeeze(0).cpu(), cls_emb_path)
            torch.save(token_embeddings.squeeze(0).cpu(), tok_emb_path)

            metadata[protein_id] = {
                'label_int': label_int, 'label_str': label_str,
                'organism_type': organism_type.strip(),
                'cls_emb_path': cls_emb_path, 'tok_emb_path': tok_emb_path
            }
        except Exception as e:
            print(f"Error processing record {record.id}: {e}. Skipping.")

    # 5. Save metadata
    metadata_filename = os.path.basename(input_fasta).split('.')[0]
    with open(os.path.join(output_dir, f'{metadata_filename}_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    # Update label_map if it was newly created
    if new_map_created:
        with open(label_map_path, 'w') as f:
            json.dump(label_map, f, indent=4)

    print(f"--- Finished processing {input_fasta} with {model_short_name} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute protein embedding features from FASTA files, supporting multiple language models.")
    parser.add_argument("--train_fasta", type=str, required=True, help="Path to the training set FASTA file.")
    parser.add_argument("--test_fasta", type=str, required=True, help="Path to the test set FASTA file.")
    parser.add_argument("--output_dir", type=str, default="./precomputed_features", help="Output directory.")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Path to the label map (globally shared)
    label_map_main_path = os.path.join(args.output_dir, 'label_map.json')

    # 🔑 List of supported models
    model_list = [
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
    ]

    for model_name in model_list:
        print(f"\n===== Running model {model_name} =====")
        process_fasta(args.train_fasta, args.output_dir, model_name, args.max_len, label_map_path=label_map_main_path)
        process_fasta(args.test_fasta, args.output_dir, model_name, args.max_len, label_map_path=label_map_main_path)
