import torch
import torch.nn as nn


class RankModel(nn.Module):
    def __init__(
        self,
        dense_dim,
        num_query_intent=5,
        num_doc_type=3,
        emb_dim=4,
        hidden_dim=32
    ):
        super().__init__()
        self.query_intent_emb = nn.Embedding(num_query_intent, emb_dim)
        self.doc_type_emb = nn.Embedding(num_doc_type, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dense_dim + emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, dense_feat, query_intent, doc_type):
        q_emb = self.query_intent_emb(query_intent)
        d_emb = self.doc_type_emb(doc_type)
        x = torch.cat([dense_feat, q_emb, d_emb], dim=-1)
        score = self.mlp(x)
        return score.squeeze(-1)
