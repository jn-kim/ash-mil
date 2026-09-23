"""
Patient-level train/val/test split of NIH ChestX-ray8.

Images are kept if they carry at least one of the 8 CXR8 disease labels or are
exactly 'No Finding'; the remaining images are grouped by Patient ID, shuffled
with `seed`, and assigned to train/val/test by cumulative image fraction.
Writes the split CSVs (file_name, labels) plus split_meta.json.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .cxr8_wsod import CLASSES_8

SPLIT_CSV_NAMES = {
    "train": "train_list_detection_811_train.csv",
    "val": "train_list_detection_811_val.csv",
    "test": "train_list_detection_811_test.csv",
}


def _read_data_entry(data_entry_csv: Path) -> List[Tuple[str, str, List[str]]]:
    """Returns [(file_name, patient_id, labels)] after label filtering."""
    if not data_entry_csv.exists():
        raise FileNotFoundError(
            f"Data_Entry_2017.csv not found: {data_entry_csv}. "
            "Pass --data_entry_csv or use --split_mode csv."
        )
    rows = []
    with data_entry_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw = set(r["Finding Labels"].split("|"))
            labels = [c for c in CLASSES_8 if c in raw]
            if not labels:
                if raw == {"No Finding"}:
                    labels = ["No Finding"]
                else:
                    continue
            rows.append((r["Image Index"].strip(), r["Patient ID"].strip(), labels))
    if not rows:
        raise ValueError(f"No usable rows in {data_entry_csv}")
    return rows


def make_patient_split(
    data_entry_csv: Path,
    seed: int = 42,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> Dict[str, List[Tuple[str, List[str]]]]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must be 3 numbers summing to 1, got {ratios}")

    rows = _read_data_entry(Path(data_entry_csv))
    by_patient: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    for file_name, pid, labels in rows:
        by_patient[pid].append((file_name, labels))

    patients = sorted(by_patient, key=lambda x: (len(x), x))  # sort so the order does not depend on dict insertion
    rng = np.random.default_rng(int(seed))
    order = [patients[i] for i in rng.permutation(len(patients))]

    total = len(rows)
    bounds = (ratios[0], ratios[0] + ratios[1])
    out: Dict[str, List[Tuple[str, List[str]]]] = {"train": [], "val": [], "test": []}
    acc = 0
    for pid in order:
        frac = acc / total
        name = "train" if frac < bounds[0] else "val" if frac < bounds[1] else "test"
        out[name].extend(by_patient[pid])
        acc += len(by_patient[pid])

    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def write_split_csvs(
    split: Dict[str, List[Tuple[str, List[str]]]],
    out_dir: Path,
    *,
    seed: int,
    ratios: Sequence[float],
    data_entry_csv: Path,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, csv_name in SPLIT_CSV_NAMES.items():
        p = out_dir / csv_name
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "labels"])
            for file_name, labels in split[name]:
                w.writerow([file_name, str(labels)])
        paths[name] = p
    meta = {
        "seed": int(seed),
        "ratios": [float(r) for r in ratios],
        "data_entry_csv": str(data_entry_csv),
        "classes": list(CLASSES_8) + ["No Finding"],
        "num_images": {k: len(v) for k, v in split.items()},
        "num_patients": {k: len({fn.split("_")[0] for fn, _ in v}) for k, v in split.items()},
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    return paths


def prepare_auto_split(
    data_entry_csv: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> Path:
    """Generate (or reuse) split CSVs under out_dir and return out_dir."""
    out_dir = Path(out_dir)
    meta_p = out_dir / "split_meta.json"
    if meta_p.exists() and all((out_dir / n).exists() for n in SPLIT_CSV_NAMES.values()):
        meta = json.loads(meta_p.read_text())
        if int(meta.get("seed", -1)) == int(seed) and [float(r) for r in meta.get("ratios", [])] == [float(r) for r in ratios]:
            print(f"[split] reusing existing split in {out_dir} (seed={seed})")
            return out_dir
        print(f"[split] existing split in {out_dir} has different seed/ratios; regenerating")

    split = make_patient_split(data_entry_csv, seed=seed, ratios=ratios)
    write_split_csvs(split, out_dir, seed=seed, ratios=ratios, data_entry_csv=Path(data_entry_csv))
    n = {k: len(v) for k, v in split.items()}
    print(f"[split] generated patient-level split (seed={seed}, ratios={tuple(ratios)}): {n} -> {out_dir}")
    return out_dir
