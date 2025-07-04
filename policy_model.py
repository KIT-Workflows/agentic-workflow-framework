import torch
import torch.nn as nn

class TripleScoringModel(nn.Module):
    def __init__(self, vocab_size, predicate_size, embedding_dim=32):
        super().__init__()
        self.entity_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.predicate_embedding = nn.Embedding(predicate_size, embedding_dim)

        # ⬇️ Scoring layer: [s | p | o] → score
        self.fc = nn.Linear(embedding_dim * 3, 1)

    def forward(self, triple_ids):
        # Ensure the input is 2D: [batch_size, 3]
        if triple_ids.ndim == 1:
            triple_ids = triple_ids.unsqueeze(0)

        subj = triple_ids[:, 0]
        pred = triple_ids[:, 1]
        obj = triple_ids[:, 2]

        s_emb = self.entity_embedding(subj)
        p_emb = self.predicate_embedding(pred)
        o_emb = self.entity_embedding(obj)

        triple_repr = torch.cat([s_emb, p_emb, o_emb], dim=1)
        return self.fc(triple_repr).squeeze()

