import json
import torch
from torch.utils.data import Dataset

from features import build_dense_features, encode_query_intent, encode_doc_type


class RankingDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]

        dense_feat = build_dense_features(x)
        query_intent = encode_query_intent(x.get("query_intent"))
        doc_type = encode_doc_type(x.get("doc_type"))
        label = float(x["label"])

        return {
            "query_id": x["query_id"],
            "doc_id": x["doc_id"],
            "dense_feat": torch.tensor(dense_feat, dtype=torch.float32),
            "query_intent": torch.tensor(query_intent, dtype=torch.long),
            "doc_type": torch.tensor(doc_type, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }
