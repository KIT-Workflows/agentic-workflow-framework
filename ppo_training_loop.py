import os
import json
import torch
import pandas as pd
from tqdm import trange
from dotenv import load_dotenv
from openai import OpenAI

from utils import generate_prompt_from_triples, compute_reward, generate_qe_input_from_prompt
from policy_model import TripleScoringModel
from sample_triples import select_topk_triples

# === Load API Key ===
load_dotenv()
openai_client = OpenAI()

# === Hyperparameters ===
EPISODES = 50
TOP_K = 5
TEMPERATURE = 0.2
INSTRUCTION = "Write a scf calculation using mixing_beta = 0.6 for the given geometry."
TEMPLATE_PATH = "prompt_template.txt"
ENTROPY_COEFF = 0.01
INITIAL_LR = 0.01
MIN_LR = 1e-4
CURRICULUM_PHASE = 10

# === Load Ground Truth .in File ===
with open("pw.scf.si.in") as f:
    ground_truth_input = f.read()

# === Load Triples ===
with open("relationships.json") as f:
    all_triples = json.load(f)

# === Show available predicates
available_predicates = sorted(set(t['predicate'] for t in all_triples))
print("Available predicates in KG:", available_predicates)

# === Build Vocab ===
entities = set()
predicates = set()
for t in all_triples:
    entities.add(t['subject'])
    entities.add(t['object'])
    predicates.add(t['predicate'])

entity2id = {e: i for i, e in enumerate(sorted(entities))}
predicate2id = {p: i for i, p in enumerate(sorted(predicates))}

# === Initialize Policy Model ===
vocab_size = len(entity2id)
predicate_size = len(predicate2id)
model = TripleScoringModel(vocab_size=vocab_size, predicate_size=predicate_size, embedding_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, min_lr=MIN_LR)

# === PPO Optimization Loop ===
results = []
for episode in trange(1, EPISODES + 1, desc="PPO Episodes"):
    # === Curriculum Phase ===
    if episode <= CURRICULUM_PHASE:
        train_triples = [t for t in all_triples if t['predicate'] == 'contains']
        print(f"Episode {episode}: {len(train_triples)} 'contains' triples selected for curriculum")
    else:
        train_triples = all_triples

    # === Triple Selection ===
    top_triples, top_ids = select_topk_triples(model, train_triples, k=TOP_K,
                                               entity2id=entity2id, predicate2id=predicate2id, return_ids=True)

    if not top_ids:
        print(f"[!] No triples selected in episode {episode}. Skipping...")
        continue  # skip to next episode

    # === Prompt and Generation ===
    prompt = generate_prompt_from_triples(top_triples, instruction=INSTRUCTION, template_path=TEMPLATE_PATH)
    generated_input = generate_qe_input_from_prompt(prompt, openai_client, temperature=TEMPERATURE)
    reward = compute_reward(generated_input, ground_truth_input)

    # === Compute Loss ===
    triple_tensor = torch.tensor(top_ids, dtype=torch.long)
    scores = model(triple_tensor)

    log_probs = torch.log_softmax(scores, dim=0)
    policy_loss = -reward * torch.sum(log_probs)
    entropy = -torch.sum(log_probs * torch.exp(log_probs))
    loss = policy_loss - ENTROPY_COEFF * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(reward)

    results.append({
        "episode": episode,
        "reward": reward,
        "triples_used": len(top_triples),
        "loss": loss.item(),
        "entropy": entropy.item(),
        "lr": optimizer.param_groups[0]['lr']
    })

# === Save Results ===
df = pd.DataFrame(results)
df.to_csv("ppo_kg_rewards.csv", index=False)
print("\n✅ PPO training completed. Reward log saved to ppo_kg_rewards.csv")

