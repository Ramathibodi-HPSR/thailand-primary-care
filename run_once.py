# run_precompute.py
from pathlib import Path
import pandas as pd
from app import precompute_coverage  # หรือชื่อไฟล์จริงของฟังก์ชัน
from app import RADII_KM             # ถ้ามีพารามิเตอร์ radii

def main():
    print("Starting precompute_coverage ...", flush=True)
    df_cov = precompute_coverage(RADII_KM)

    out_path = Path("data") / "coverage.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_cov.to_parquet(out_path)
    print("Done. Saved coverage to", out_path, flush=True)

if __name__ == "__main__":
    main()
