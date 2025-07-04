import torch

def select_topk_triples(model, triples, k, entity2id, predicate2id, return_ids=False):
    encoded = []
    valid_triples = []
    for t in triples:
        try:
            s = entity2id[t['subject']]
            p = predicate2id[t['predicate']]
            o = entity2id[t['object']]
            encoded.append([s, p, o])
            valid_triples.append(t)
        except KeyError:
            continue  # skip unknown entities/predicates

    if not encoded:
        return ([], []) if return_ids else []

    triple_tensor = torch.tensor(encoded, dtype=torch.long)
    with torch.no_grad():
        scores = model(triple_tensor)

    topk_indices = torch.topk(scores, k=min(k, len(encoded))).indices.tolist()
    top_triples = [valid_triples[i] for i in topk_indices]
    top_ids = [encoded[i] for i in topk_indices]

    if return_ids:
        return top_triples, top_ids
    else:
        return top_triples

