import os
import json
import pickle
from typing import List, Tuple, Dict

import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel


def load_target_concepts(ontology_file: str, target_ids_file: str) -> Dict[str, str]:
    """Load target concept names and synonyms from the ontology."""
    with open(ontology_file, "r") as f:
        ontology = json.load(f)
    with open(target_ids_file, "r") as f:
        target_ids = set(json.load(f))

    name_map = {}
    for concept_id, entry in ontology.items():
        if concept_id not in target_ids:
            continue
        names = list(set([entry[0]] + entry[-1]))  # main name + synonyms
        for i, name in enumerate(names):
            name_map[f"{concept_id}/{i}"] = name
    print(f"Loaded {len(name_map)} names/synonyms from ontology.")
    return name_map


def embed_names(name_map: Dict[str, str], model, tokenizer, device: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """Compute mean-pooled embeddings for concept names and synonyms."""
    entries = list(name_map.items())
    inputs = [{'input_ids': tokenizer.encode(name, max_length=128, truncation=True)} for _, name in entries]

    dataset = Dataset.from_list(inputs)

    def collate_fn(batch):
        input_ids = [x['input_ids'] + [tokenizer.pad_token_id] * (128 - len(x['input_ids'])) for x in batch]
        attention_mask = [[1] * len(x['input_ids']) + [0] * (128 - len(x['input_ids'])) for x in batch]
        return torch.tensor(input_ids), torch.tensor(attention_mask)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model.eval()
    model.to(device)

    all_embeddings, idx_list, label_list = [], [], []

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Embedding names"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            masked_embeddings = hidden_states * mask

            summed = masked_embeddings.sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            mean_pooled = summed / counts
            mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            all_embeddings.extend(mean_pooled.cpu().numpy())

    idx_list = [e[0] for e in entries]
    label_list = [e[1] for e in entries]
    return idx_list, np.array(all_embeddings), label_list


def save_embeddings(output_path: str, idx_list: List[str], embeddings: np.ndarray, labels: List[str]):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump([idx_list, embeddings, labels], f)
    print(f"Saved {len(idx_list)} embeddings to: {output_path}")


def main():
    # Settings
    ontology_file = "../data/onto/hoip/concept_info.json"
    target_ids_file = "../data/onto/hoip/target_concept_id_list.json"
    embedding_model = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    output_file = "models/hoip/embedding/ontology_concept_embeddings.pkl"
    device = "cpu"  # "cuda" or "cpu" depending on your system

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)

    # Load target concept names
    name_map = load_target_concepts(ontology_file, target_ids_file)

    # Compute embeddings
    idx_list, embeddings, labels = embed_names(name_map, model, tokenizer, device)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Save output
    save_embeddings(output_file, idx_list, embeddings, labels)


if __name__ == "__main__":
    main()
