import os
import json
import argparse
from pathlib import Path
from openai import OpenAI
import yaml
import csv

from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

def summarize_markdown(content):
    prompt = f"Summarize this Markdown file in 2–3 sentences:\n\n{content[:2000]}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def extract_relationships(content):
    prompt = (
        "Extract factual relationships as subject–predicate–object triples from this content. "
        "Return ONLY a JSON array of objects with 'subject', 'predicate', and 'object' keys.\n\n"
        f"{content[:3000]}"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        print(f"[!] Failed to parse JSON from model response:\n{response.choices[0].message.content}\n")
        return []

def convert_dict_to_triples(source, data):
    triples = []
    for subject, nested in data.items():
        if isinstance(nested, dict):
            for predicate, objects in nested.items():
                if isinstance(objects, list):
                    for obj in objects:
                        triples.append({"subject": subject, "predicate": predicate, "object": obj, "source": source})
        elif isinstance(nested, list):
            for obj in nested:
                triples.append({"subject": subject, "predicate": "is_a", "object": obj, "source": source})
    return triples

def extract_triples_from_parameter_list(source, data):
    triples = []
    for entry in data:
        subject = entry.get("Parameter_Name")
        if not subject:
            continue
        final_comment = entry.get("Final_comments") or ""
        if "deprecated" in final_comment.lower():
            continue  # Skip deprecated entries
        namelist = entry.get("Namelist", "unknown")
        for key, value in entry.items():
            if key == "Parameter_Name":
                continue
            if isinstance(value, (str, int, float, bool)):
                triples.append({
                    "subject": subject,
                    "predicate": key,
                    "object": value,
                    "source": source,
                    "namelist": namelist
                })
            elif isinstance(value, dict):
                for subkey, subval in value.items():
                    triples.append({
                        "subject": subject,
                        "predicate": f"{key}:{subkey}",
                        "object": subval,
                        "source": source,
                        "namelist": namelist
                    })
            elif isinstance(value, list):
                for item in value:
                    triples.append({
                        "subject": subject,
                        "predicate": key,
                        "object": item,
                        "source": source,
                        "namelist": namelist
                    })
    return triples

def process_directory(source_dir: Path, filetypes: set):
    document_nodes = []
    relationships = []

    for file in source_dir.iterdir():
        if file.suffix in [".md", ".mdx"] and "md" in filetypes:
            print(f"\U0001F4C4 Processing {file.name}")
            content = file.read_text()

            try:
                summary = summarize_markdown(content)
                document_nodes.append({
                    "id": file.name,
                    "title": file.stem,
                    "summary": summary,
                })
            except Exception as e:
                print(f"[!] Skipping summary for {file.name}: {e}")

            try:
                triples = extract_relationships(content)
                for t in triples:
                    relationships.append({
                        "subject": t["subject"],
                        "predicate": t["predicate"],
                        "object": t["object"],
                        "source": file.name,
                        "namelist": "markdown"
                    })
            except Exception as e:
                print(f"[!] Skipping relationships for {file.name}: {e}")

        elif file.suffix == ".json" and "json" in filetypes:
            print(f"\U0001F4E6 Loading JSON file: {file.name}")
            try:
                with open(file) as f:
                    data = json.load(f)

                if file.name == "xqe_univ_kg_load_v1.json":
                    print("🛠 Forcing parameter list parser")
                    triples = extract_triples_from_parameter_list(file.name, data)
                    relationships.extend(triples)
                    print(f"✅ Extracted {len(triples)} triples from {file.name}")
                    continue

                if isinstance(data, list):
                    if all(isinstance(d, dict) and all(k in d for k in ("subject", "predicate", "object")) for d in data):
                        relationships.extend(data)
                    elif all(isinstance(d, dict) and all(k in d for k in ("id", "title", "summary")) for d in data):
                        document_nodes.extend(data)
                    elif all(isinstance(d, dict) and "Parameter_Name" in d for d in data):
                        print(f"\U0001F527 Detected parameter list in {file.name}")
                        triples = extract_triples_from_parameter_list(file.name, data)
                        relationships.extend(triples)
                        print(f"✅ Extracted {len(triples)} triples from {file.name}")
                    else:
                        print(f"[!] Unknown JSON list structure in {file.name}")
                elif isinstance(data, dict):
                    print(f"\U0001F501 Attempting conversion from nested dict JSON in {file.name}")
                    triples = convert_dict_to_triples(file.name, data)
                    relationships.extend(triples)
                    print(f"✅ Converted {len(triples)} triples from {file.name}")
                else:
                    print(f"[!] Unknown JSON structure in {file.name}")
            except Exception as e:
                print(f"[!] Failed to load JSON from {file.name}: {e}")

        elif file.suffix in [".yaml", ".yml"] and "yaml" in filetypes:
            print(f"\U0001F499 Loading YAML file: {file.name}")
            try:
                with open(file) as f:
                    data = yaml.safe_load(f)

                if isinstance(data, list):
                    if all(isinstance(d, dict) and all(k in d for k in ("subject", "predicate", "object")) for d in data):
                        relationships.extend(data)
                    elif all(isinstance(d, dict) and all(k in d for k in ("id", "title", "summary")) for d in data):
                        document_nodes.extend(data)
                    else:
                        print(f"[!] Unknown YAML list structure in {file.name}")
                elif isinstance(data, dict):
                    triples = convert_dict_to_triples(file.name, data)
                    relationships.extend(triples)
                    print(f"✅ Converted {len(triples)} triples from {file.name}")
                else:
                    print(f"[!] Unknown YAML structure in {file.name}")
            except Exception as e:
                print(f"[!] Failed to load YAML from {file.name}: {e}")

    return document_nodes, relationships

def main():
    parser = argparse.ArgumentParser(description="Extract KG data from Markdown, JSON, and/or YAML files.")
    parser.add_argument("--source", "-s", type=str, default="markdown_files", help="Path to the source folder.")
    parser.add_argument("--types", "-t", type=str, default="all", choices=["md", "json", "yaml", "all"],
                        help="Which file types to process: md, json, yaml, or all.")
    parser.add_argument("--out-docs", default="document_nodes.json", help="Output path for document nodes.")
    parser.add_argument("--out-rels", default="relationships.json", help="Output path for relationships.")

    args = parser.parse_args()
    source_dir = Path(args.source)
    filetypes = {"md", "json", "yaml"} if args.types == "all" else {args.types}

    document_nodes, relationships = process_directory(source_dir, filetypes)

    # Deduplicate
    unique_relationships = {
        json.dumps(rel, sort_keys=True): rel for rel in relationships
    }
    relationships = list(unique_relationships.values())
    print(f"✅ Deduplicated to {len(relationships)} unique relationships")

    with open(args.out_docs, "w") as f:
        json.dump(document_nodes, f, indent=2)
    print(f"✅ Saved {len(document_nodes)} documents to {args.out_docs}")

    with open(args.out_rels, "w") as f:
        json.dump(relationships, f, indent=2)
    print(f"✅ Saved {len(relationships)} relationships to {args.out_rels}")

    # Export to CSV
    with open("relationships.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "predicate", "object", "source", "namelist"])
        writer.writeheader()
        writer.writerows(relationships)
    print("✅ Exported relationships to relationships.csv")

if __name__ == "__main__":
    main()
