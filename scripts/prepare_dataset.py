"""
Dataset audit, validation, cleaning, and leakage-safe stratification.
Inspects raw images, detects corrupt/zero-byte files, resolves cross-identity duplicates,
normalizes player names, and generates reproducible train/val/test manifests.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import csv
from PIL import Image

# Name normalization mapping for scraped directory artifacts
SCRAPER_DIGIT_MAP = {
    'Anil Kumble1': 'Anil Kumble',
    'Ashish Nehra1': 'Ashish Nehra',
    'Harbhajan Singh1': 'Harbhajan Singh',
    'M.S. Dhoni1': 'M.S. Dhoni',
    'Rahul Dravid2': 'Rahul Dravid',
    'Sourav Ganguly1': 'Sourav Ganguly',
    'Virender Sehwag2': 'Virender Sehwag',
    'Yuvraj Singh1': 'Yuvraj Singh',
    'Zaheer Khan1': 'Zaheer Khan',
}

def clean_player_name(folder_name: str) -> str:
    """Normalize player folder name, stripping scraping artifacts."""
    return SCRAPER_DIGIT_MAP.get(folder_name, folder_name).strip()

def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def is_all_null_bytes(path: Path, max_check: int = 65536) -> bool:
    """Check if the beginning of the file is purely null bytes."""
    with open(path, 'rb') as f:
        data = f.read(max_check)
        if not data or set(data) == {0}:
            # Verify rest of file
            f.seek(0)
            while chunk := f.read(65536):
                if set(chunk) != {0}:
                    return False
            return True
    return False

def audit_and_prepare_dataset(
    raw_data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    import random
    random.seed(random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Auditing dataset in: {raw_data_dir}")
    player_folders = sorted([f for f in raw_data_dir.iterdir() if f.is_dir()])
    print(f"Found {len(player_folders)} player folders.")
    
    audit_results = {
        "total_files_scanned": 0,
        "valid_files": 0,
        "excluded_files": {},
        "exclusion_reasons": Counter(),
        "cleaned_identities_count": 0,
        "cross_identity_duplicates_count": 0,
        "same_identity_duplicates_count": 0,
    }
    
    # 1. First pass: Scan all files, check integrity, compute hashes
    hash_to_files = defaultdict(list)
    file_records = []
    
    for folder in player_folders:
        folder_name = folder.name
        norm_name = clean_player_name(folder_name)
        
        for file_path in folder.iterdir():
            if not file_path.is_file():
                continue
                
            audit_results["total_files_scanned"] += 1
            rel_path = str(file_path.relative_to(raw_data_dir))
            
            # Check zero-byte
            sz = file_path.stat().st_size
            if sz == 0:
                audit_results["excluded_files"][rel_path] = "Zero-byte file"
                audit_results["exclusion_reasons"]["zero_byte"] += 1
                continue
                
            # Check null bytes
            if is_all_null_bytes(file_path):
                audit_results["excluded_files"][rel_path] = "Corrupt file: 100% null (0x00) bytes"
                audit_results["exclusion_reasons"]["corrupt_null_bytes"] += 1
                continue
                
            # Check PIL decodability & dimensions
            try:
                with Image.open(file_path) as im:
                    w, h = im.size
                    mode = im.mode
                    if w < 50 or h < 50:
                        audit_results["excluded_files"][rel_path] = f"Low resolution: {w}x{h} < 50px"
                        audit_results["exclusion_reasons"]["low_resolution"] += 1
                        continue
            except Exception as e:
                audit_results["excluded_files"][rel_path] = f"Unreadable image: {str(e)}"
                audit_results["exclusion_reasons"]["unreadable"] += 1
                continue
                
            file_hash = compute_file_hash(file_path)
            hash_to_files[file_hash].append({
                "path": file_path,
                "rel_path": rel_path,
                "raw_folder": folder_name,
                "norm_identity": norm_name,
                "size": sz,
                "dimensions": (w, h),
                "mode": mode,
                "hash": file_hash
            })

    # 2. Duplicate Resolution
    # Separate into same-identity duplicates and cross-identity duplicates
    retained_records = []
    
    for file_hash, records in hash_to_files.items():
        if len(records) == 1:
            retained_records.append(records[0])
            continue
            
        # Multiple files with identical SHA-256
        identities = set(r["norm_identity"] for r in records)
        if len(identities) > 1:
            # Cross-identity conflict: Contradictory labels!
            # Exclude all instances to eliminate label poisoning
            audit_results["cross_identity_duplicates_count"] += len(records)
            for r in records:
                msg = f"Cross-identity duplicate conflict between: {sorted(identities)}"
                audit_results["excluded_files"][r["rel_path"]] = msg
                audit_results["exclusion_reasons"]["cross_identity_conflict"] += 1
        else:
            # Same identity duplicate: Keep the first one, exclude duplicates
            audit_results["same_identity_duplicates_count"] += (len(records) - 1)
            retained_records.append(records[0])
            for r in records[1:]:
                msg = f"Same-identity exact duplicate of {records[0]['rel_path']}"
                audit_results["excluded_files"][r["rel_path"]] = msg
                audit_results["exclusion_reasons"]["same_identity_duplicate"] += 1

    audit_results["valid_files"] = len(retained_records)
    print(f"Total scanned: {audit_results['total_files_scanned']}")
    print(f"Total excluded: {len(audit_results['excluded_files'])}")
    print(f"Total clean retained files: {audit_results['valid_files']}")
    print(f"Exclusion breakdown: {dict(audit_results['exclusion_reasons'])}")

    # Group by clean identity
    by_identity = defaultdict(list)
    for r in retained_records:
        by_identity[r["norm_identity"]].append(r)
        
    audit_results["cleaned_identities_count"] = len(by_identity)
    print(f"Clean distinct identities: {len(by_identity)}")
    
    # 3. Stratified Split (Train 70%, Val 15%, Test 15%)
    train_records = []
    val_records = []
    test_records = []
    
    for identity, items in sorted(by_identity.items()):
        random.shuffle(items)
        n = len(items)
        if n >= 6:
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * test_ratio)))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = n - 2
                n_val = 1
                n_test = 1
        elif n >= 3:
            n_train = n - 2
            n_val = 1
            n_test = 1
        elif n == 2:
            n_train = 1
            n_val = 1
            n_test = 0
        else:
            n_train = 1
            n_val = 0
            n_test = 0
            
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]
        
        for item in train_items:
            item["split"] = "train"
            train_records.append(item)
        for item in val_items:
            item["split"] = "val"
            val_records.append(item)
        for item in test_items:
            item["split"] = "test"
            test_records.append(item)

    print(f"Stratified Splits: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    
    # 4. Write CSV Manifests
    def write_csv(manifest_path: Path, records: list):
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["path", "identity", "raw_folder", "split", "width", "height", "mode", "sha256"]
            )
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "path": r["rel_path"].replace('\\', '/'),
                    "identity": r["norm_identity"],
                    "raw_folder": r["raw_folder"],
                    "split": r["split"],
                    "width": r["dimensions"][0],
                    "height": r["dimensions"][1],
                    "mode": r["mode"],
                    "sha256": r["hash"]
                })

    all_records = train_records + val_records + test_records
    write_csv(output_dir / "manifest_clean.csv", all_records)
    write_csv(output_dir / "manifest_train.csv", train_records)
    write_csv(output_dir / "manifest_val.csv", val_records)
    write_csv(output_dir / "manifest_test.csv", test_records)

    # 5. Write Audit Report
    audit_report_path = output_dir / "audit_report.json"
    with open(audit_report_path, 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2)

    # Write Markdown summary
    md_path = output_dir / "audit_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Audit and Preprocessing Report\n\n")
        f.write(f"- **Total files scanned**: {audit_results['total_files_scanned']}\n")
        f.write(f"- **Total files excluded**: {len(audit_results['excluded_files'])}\n")
        f.write(f"- **Total clean files retained**: {audit_results['valid_files']}\n")
        f.write(f"- **Identities represented**: {len(by_identity)}\n")
        f.write(f"- **Stratification**: Train = {len(train_records)}, Val = {len(val_records)}, Test = {len(test_records)}\n\n")
        f.write("## Exclusion Reasons\n\n")
        for reason, count in audit_results["exclusion_reasons"].items():
            f.write(f"- `{reason}`: {count}\n")
        f.write("\n## Excluded Files Detail\n\n")
        f.write("| File Path | Reason |\n|---|---|\n")
        for pth, reason in sorted(audit_results["excluded_files"].items()):
            f.write(f"| `{pth}` | {reason} |\n")

    print(f"Generated manifests and reports in {output_dir}")
    return audit_results

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = base_dir / "data" / "archive (3)" / "indian cricketer"
    out_dir = base_dir / "data"
    audit_and_prepare_dataset(raw_dir, out_dir)

