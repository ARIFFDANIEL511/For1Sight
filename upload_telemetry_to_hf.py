"""For1Sight - one-off upload of the Silver race telemetry to a Hugging Face dataset.

Uploads hf_upload/race_telemetry/year=*/event_slug=*/*.parquet (exported from HDFS)
to https://huggingface.co/datasets/<your-username>/for1sight-race-telemetry,
one race partition per commit. Partitions already on Hugging Face are skipped,
so if the upload stops, just run this script again to continue.

Run from the project folder:  python upload_telemetry_to_hf.py
"""
from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi

REPO_NAME = "for1sight-race-telemetry"
PRIVATE = False  # public dataset -> the deployed app needs no token
SILVER_ROOT = Path(__file__).resolve().parent / "hf_upload" / "race_telemetry"

api = HfApi()
username = api.whoami()["name"]
repo_id = f"{username}/{REPO_NAME}"
print(f"huggingface_hub {huggingface_hub.__version__} | dataset {repo_id}")

api.create_repo(repo_id, repo_type="dataset", private=PRIVATE, exist_ok=True)

# Which partitions are already uploaded (allows resuming)
existing = {
    str(Path(p).parent).replace("\\", "/")
    for p in api.list_repo_files(repo_id, repo_type="dataset")
    if p.endswith(".parquet")
}

partitions = sorted(p for p in SILVER_ROOT.glob("year=*/event_slug=*") if p.is_dir())
total_gb = sum(f.stat().st_size for f in SILVER_ROOT.rglob("*.parquet")) / 1024**3
print(f"{len(partitions)} race partitions, {total_gb:.2f} GB in total\n")

for number, folder in enumerate(partitions, start=1):
    path_in_repo = f"race_telemetry/{folder.parent.name}/{folder.name}"
    label = f"[{number}/{len(partitions)}] {path_in_repo}"
    if path_in_repo in existing:
        print(f"{label}  already uploaded - skipped")
        continue
    size_mb = sum(f.stat().st_size for f in folder.glob("*.parquet")) / 1024**2
    print(f"{label}  uploading {size_mb:.0f} MB ...", flush=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        allow_patterns=["*.parquet"],
        commit_message=f"Add Silver race telemetry {folder.parent.name} {folder.name}",
    )

print("\nUpload complete.")
print(f"Dataset page : https://huggingface.co/datasets/{repo_id}")
print(f'Use this value: FOR1SIGHT_HF_DATASET = "{repo_id}"')
