from pathlib import Path
from collections import defaultdict

# Define your folders
T1_DIR     = Path("Dataset/Images/T1W-TSE")
T2_DIR     = Path("Dataset/Images/STIR")
MASKS_DIR  = Path("Dataset/Masks")
LESION_DIR = Path("Dataset/Labels")

# Containers for results
missing_files = defaultdict(list)  # pid -> list of missing (label, path)
correct_count = 0
corrupted_count = 0

# Loop over each T1 scan
for t1_path in sorted(T1_DIR.glob("Patient_*.nii.gz")):
    pid = t1_path.name.replace(".nii.gz", "")  # e.g. "Patient_20"
    num = int(pid.split("_")[1])

    # Expected files
    counterparts = {
        "T1":     T1_DIR     / f"{pid}.nii.gz",
        "T2":     T2_DIR     / f"{pid}.nii.gz",
        "Mask":   MASKS_DIR  / f"BOT_{num:03d}.nii.gz",
        "Lesion": LESION_DIR / f"{pid}.nii.gz",
    }

    # Check existence
    this_missing = []
    for label, path in counterparts.items():
        if not path.exists():
            this_missing.append((label, str(path)))

    # Tally results
    if not this_missing:
        correct_count += 1
    else:
        corrupted_count += 1
        missing_files[pid].extend(this_missing)

# Print summary
total = correct_count + corrupted_count
print(f"Total patients checked: {total}")
print(f"  Complete cases: {correct_count}")
print(f"  Incomplete cases: {corrupted_count}")

# Print details of missing files
if missing_files:
    print("\nMissing files by patient:")
    for pid, misses in missing_files.items():
        print(f"{pid}:")
        for label, path in misses:
            print(f"  - {label}: {path}")
else:
    print("\nNo missing files detected.")