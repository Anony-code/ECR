import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

# path = Path("data/test_fold0_ratio10_neg5_visit2.jsonl") # AD
# path = Path("data/PD_test_fold0_ratio5.jsonl") # PD
# path = Path("data/ADRD_test_fold0_ratio10.jsonl") # PD
path = Path("data/PD_test_fold0_ratio5_neg3.jsonl") # PD_jan29


records = []
labels = []
with path.open() as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)
            records.append(obj)
            labels.append(obj["label"])

print("Helper: get records from test data", len(records))
print("cwd:", Path.cwd())
print("input exists:", path.exists(), path.resolve())
print("label counts:", {0: labels.count(0), 1: labels.count(1)})


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
out_dir = Path("data/PDstratified5/")
# out_dir = Path("data/ADRDstratified10/")

out_dir.mkdir(parents=True, exist_ok=True)
print("Helper: saving folds to", out_dir.resolve())

for fold_idx, (_, test_idx) in enumerate(skf.split(records, labels)):

    out_path = out_dir / f"test_fold{fold_idx:01d}.jsonl"
    pos_count = 0
    neg_count = 0
    with out_path.open("w") as f:
        for i in test_idx:
            label = records[i]["label"]
            if label == 1:
                pos_count += 1
            else:
                neg_count += 1
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")

    print(f"fold {fold_idx:01d}: pos={pos_count}, neg={neg_count}")

