'''compiling all bad channels across all subjects as a csv'''
import json
import pandas as pd
from pathlib import Path
import numpy as np

Params_dir = Path("/home/calm-brain/vsuresh/eeg_preprocessing/output_files/subject_params")
Output_dir = Path("/home/calm-brain/vsuresh/eeg_feature_extraction/cohort-level metrics")
Output_dir.mkdir(parents=True, exist_ok=True)

Output_csv = Output_dir / "all_subjects_bad_channels.csv"
Output_csv_long = Output_dir / "all_subjects_bad_channels_long.csv"

rows = []
rows_long = []

for json_path in sorted(Params_dir.glob("*.json")):
    with open(json_path, "r") as f:
        data = json.load(f)

    subject_id = data.get("subject_id", json_path.stem)
    bad_channels = data.get("bad_channels", [])

    rows.append({
        "subject_id": subject_id,
        "n_bad_channels": len(bad_channels),
        "bad_channels": ";".join(bad_channels),
    })

    for ch in bad_channels:
        rows_long.append({"subject_id": subject_id, "bad_channel": ch})

df = pd.DataFrame(rows)
df.to_csv(Output_csv, index=False)
print(f"Saved {len(df)} subjects to {Output_csv}")

df_long = pd.DataFrame(rows_long)
df_long.to_csv(Output_csv_long, index=False)
print(f"Saved {len(df_long)} bad-channel entries to {Output_csv_long}")