#!/usr/bin/env python3
import argparse
import csv
import getpass
import gzip
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0"
METADATA_NAME = "mimic-cxr-2.0.0-metadata.csv"


def build_opener(user: str, password: str) -> urllib.request.OpenerDirector:
    mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, BASE_URL, user, password)
    return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(mgr))


def download(opener: urllib.request.OpenerDirector, url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with opener.open(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dst)


def ensure_metadata(opener: urllib.request.OpenerDirector, out_dir: Path) -> Path:
    csv_path = out_dir / METADATA_NAME
    if csv_path.exists():
        return csv_path
    gz_path = out_dir / (METADATA_NAME + ".gz")
    print(f"Downloading {METADATA_NAME}.gz ...")
    download(opener, f"{BASE_URL}/{METADATA_NAME}.gz", gz_path)
    with gzip.open(gz_path, "rb") as src, open(csv_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink()
    return csv_path


def resolve_paths(dicom_ids: List[str], metadata_csv: Path) -> Dict[str, str]:
    want = set(dicom_ids)
    rel: Dict[str, str] = {}
    with metadata_csv.open("r", newline="") as f:
        for row in csv.DictReader(f):
            did = row.get("dicom_id")
            if did in want:
                subj, study = row["subject_id"], row["study_id"]
                rel[did] = f"files/p{subj[:2]}/p{subj}/s{study}/{did}.jpg"
                if len(rel) == len(want):
                    break
    return rel


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the MIMIC-CXR-JPG images used for held-out evaluation.")
    ap.add_argument("--ann", default="data/mimic/annotations/221v2hiqualnihsplit.json", help="Held-out COCO json.")
    ap.add_argument("--out_dir", required=True, help="Target root, e.g. $DATA_ROOT/mimic-cxr-jpg")
    ap.add_argument("--user", required=True, help="PhysioNet username.")
    ap.add_argument("--password", default=os.environ.get("PHYSIONET_PASSWORD", ""), help="PhysioNet password (or set PHYSIONET_PASSWORD; prompted if empty).")
    args = ap.parse_args()

    ann_path = Path(args.ann)
    if not ann_path.exists():
        sys.exit(f"Annotation json not found: {ann_path}. Clone https://github.com/leotam/MIMIC-CXR-annotations into data/mimic first.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    password = args.password or getpass.getpass("PhysioNet password: ")
    opener = build_opener(args.user, password)

    dicom_ids = sorted({Path(im["file_name"]).stem for im in json.loads(ann_path.read_text())["images"]})
    print(f"{len(dicom_ids)} held-out images listed in {ann_path.name}")

    metadata_csv = ensure_metadata(opener, out_dir)
    rel_paths = resolve_paths(dicom_ids, metadata_csv)
    missing = [d for d in dicom_ids if d not in rel_paths]
    if missing:
        sys.exit(f"{len(missing)} dicom_ids not found in {metadata_csv.name}. First 5: {missing[:5]}")

    todo = [(d, r) for d, r in rel_paths.items() if not (out_dir / r).exists()]
    print(f"{len(rel_paths) - len(todo)} already present, {len(todo)} to download")
    failed = []
    for did, rel in tqdm(todo, desc="Downloading", unit="img"):
        try:
            download(opener, f"{BASE_URL}/{rel}", out_dir / rel)
        except Exception as exc:
            failed.append((did, str(exc)))
    if failed:
        print(f"{len(failed)} downloads failed (re-run to retry). First 5:")
        for did, err in failed[:5]:
            print(f"  {did}: {err}")
        sys.exit(1)
    print(f"Done. Images are under {out_dir / 'files'}, metadata at {metadata_csv}")


if __name__ == "__main__":
    main()
