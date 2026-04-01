import json
from collections import Counter, defaultdict


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows


def print_data_stats(rows):
    intent_counter = Counter()
    doc_type_counter = Counter()
    label_counter = Counter()
    missing_counter = Counter()
    query_docs = defaultdict(int)

    for r in rows:
        intent_counter[r["query_intent"]] += 1
        doc_type_counter[r["doc_type"]] += 1
        label_counter[r["label"]] += 1
        query_docs[r["query_id"]] += 1

        for k, v in r.items():
            if v is None:
                missing_counter[k] += 1

    print("=== Data Stats ===")
    print("num_rows:", len(rows))
    print("intent_dist:", dict(intent_counter))
    print("doc_type_dist:", dict(doc_type_counter))
    print("label_dist:", dict(label_counter))
    print("avg_docs_per_query:", round(
        sum(query_docs.values()) / max(len(query_docs), 1), 2))
    print("missing_fields:", dict(missing_counter))
