# PPO-KG: Reinforcement Learning on Knowledge Graphs for Quantum ESPRESSO Protocol Generation

This project implements a reinforcement learning (RL) loop using Proximal Policy Optimization (PPO) to train a model that selects knowledge triples from a scientific Knowledge Graph (KG). These triples are used to condition language model prompts that generate Quantum ESPRESSO (QE) input files. The goal is to maximize the similarity between the generated QE file and a given ground truth `.in` file.

---

## 🧠 Workflow Summary

The pipeline starts from a curated knowledge graph of QE parameters. A neural policy model scores subject–predicate–object triples to identify the most useful ones for generating valid QE input files. These top-ranked triples are converted into natural language prompts using a template. The prompt is passed to a language model (via OpenAI's API), which generates a `.in` file. A reward is calculated by comparing this generated file with a reference ground truth file. PPO is then used to optimize the triple selection policy based on this reward.

---

## 📁 Project Structure

```
.
├── ppo_training_loop.py        # Main PPO optimization script
├── policy_model.py             # Triple scoring model (neural network)
├── sample_triples.py           # Top-k triple selection logic
├── utils.py                    # Prompt generation, reward computation, LM querying
├── relationships.json          # Input knowledge graph (triples)
├── prompt_template.txt         # Prompt skeleton with placeholders
├── pw.scf.si.in                # Ground truth QE input file for reward comparison
├── ppo_kg_rewards.csv          # Training logs: reward, loss, entropy, lr
├── extract_kg_data.py          # (Optional) Triple extractor from raw documents
```

---

## 🔧 Component Descriptions

### `ppo_training_loop.py`

Main script that runs PPO training across multiple episodes. It:

* Loads the ground truth input file and KG triples
* Initializes vocabulary and policy model
* Selects top-k triples using the model
* Generates a prompt and queries the OpenAI API
* Computes reward and performs PPO-based policy updates
* Logs training progress to `ppo_kg_rewards.csv`

### `policy_model.py`

Defines a PyTorch model (`TripleScoringModel`) that embeds subject, predicate, and object tokens and scores them via a feedforward layer.

### `sample_triples.py`

Implements `select_topk_triples(...)`:

* Encodes triples as index vectors
* Applies the model to rank them
* Returns the top-k triples and their IDs

### `utils.py`

* `generate_prompt_from_triples(...)`: Fills in a text template with the selected triples and instruction.
* `generate_qe_input_from_prompt(...)`: Sends prompt to OpenAI and returns generated `.in` content.
* `compute_reward(...)`: Computes line-wise overlap (Jaccard index) between generated and ground truth files.

### `relationships.json`

Knowledge graph encoded as a list of triples (`subject`, `predicate`, `object`) extracted from documentation.

### `prompt_template.txt`

Template prompt provided to the language model. Contains placeholders for `<<FACTS>>` and instructions.

### `pw.scf.si.in`

The ground truth QE input file used to compute rewards. Your model is trained to generate outputs that resemble this file.

### `extract_kg_data.py`

Utility script to extract relationships from markdown, YAML, or JSON technical documentation using GPT-based summarization and parsing.

---

## 🚀 How to Run

1. **Install dependencies** (via Conda or Pip):

   ```bash
   pip install openai torch pandas tqdm python-dotenv
   ```

2. **Set up your API key** in a `.env` file:

   ```
   OPENAI_API_KEY=sk-...
   ```

3. **Start training**:

   ```bash
   python ppo_training_loop.py
   ```

4. **Monitor progress** in `ppo_kg_rewards.csv`:

   * `reward`: Similarity between generated and ground truth `.in` file
   * `loss`: PPO loss
   * `entropy`: Diversity of the policy
   * `lr`: Current learning rate

---

## 📊 Example Output

After training, a sample `ppo_kg_rewards.csv` might look like:

| episode | reward | loss | entropy | lr       |
| ------- | ------ | ---- | ------- | -------- |
| 1       | 0.17   | 1.39 | 1.55    | 0.010000 |
| 20      | 0.20   | 1.63 | 1.60    | 0.005000 |
| 50      | 0.20   | 1.63 | 1.60    | 0.000156 |

---

## 📌 Notes

* The curriculum learning phase focuses initially on `"contains"` triples to help bootstrap training.
* Learning rate is automatically reduced based on reward plateaus (`ReduceLROnPlateau`).
* The OpenAI GPT model is used in chat mode with temperature control.

---

## 🧩 Optional Enhancements

* Integrate domain-specific scoring metrics in `compute_reward()`
* Add visualization: reward curves, triple usage frequency
* Replace OpenAI API with a local LLM for open-source deployment

---

Let me know if you’d like a Markdown version of this README saved to file.
