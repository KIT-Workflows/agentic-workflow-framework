import os

def generate_prompt_from_triples(triples, instruction, template_path="prompt_template.txt"):
    """
    Given a list of triples and a user instruction, fill in the prompt template.
    """
    with open(template_path) as f:
        template = f.read()

    facts = ""
    for t in triples:
        facts += f"{t['subject']} — {t['predicate']} — {t['object']}\n"

    return template.replace("<<FACTS>>", facts.strip()).replace("<<INSTRUCTION>>", instruction)


def compute_reward(generated_input, ground_truth_input):
    """
    Reward function based on line-level overlap.
    """
    gen_lines = set(generated_input.strip().splitlines())
    gt_lines = set(ground_truth_input.strip().splitlines())
    overlap = gen_lines.intersection(gt_lines)
    union = gen_lines.union(gt_lines)
    return len(overlap) / len(union) if union else 0.0


def generate_qe_input_from_prompt(prompt, openai_client, temperature=0.7):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[!] Prompt failed: {e}")
        return ""

