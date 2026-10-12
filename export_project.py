import os
import zipfile

ZIP_NAME = "DeepTrace_AI_Project.zip"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, ZIP_NAME)

files_to_zip = [
    "app.py",
    "common.py",
    "dataset.py",
    "knowledge_base.py",
    "predict.py",
    "train.py",
    "requirements.txt",
    "README.md",
]

dirs_to_zip = ["data", "artifacts"]

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in files_to_zip:
        fp = os.path.join(BASE_DIR, f)
        if os.path.exists(fp):
            zipf.write(fp, arcname=f)
            print(f"Added file: {f}")

    for d in dirs_to_zip:
        dp = os.path.join(BASE_DIR, d)
        if os.path.exists(dp):
            for root, _, files in os.walk(dp):
                for file in files:
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, BASE_DIR)
                    zipf.write(full_p, arcname=rel_p)
                    print(f"Added artifact: {rel_p}")

print(f"\n[SUCCESS] Project successfully exported to: {ZIP_PATH}")
