# scripts/logger.py
import csv
import os

class CSVLogger:
    def __init__(self, out_dir="results", filename="baseline.csv", fieldnames=None):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, filename)
        self.fieldnames = fieldnames or ["episode", "reward", "win", "steps"]
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, **kwargs):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(kwargs)
