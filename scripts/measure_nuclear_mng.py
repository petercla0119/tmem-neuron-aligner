"""Add nuclear mNG measurements to a completed plate using its saved image arrays."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tmem_align.analysis.nuclear_signal import quantify_nuclear_mng


def measure(task):
    root, well, day, count, scale = task
    root = Path(root)
    key = f"{well}_d{day}"
    image = root / "wells" / well / "preprocessed" / f"{key}.npz"
    mask = root / "wells" / well / "frames" / f"{key}_labels.npz"
    cache = root / "nuclear_mng" / "frames" / f"{key}.json"
    signature = {
        str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns] for p in [image, mask]
    }
    if cache.exists():
        result = json.loads(cache.read_text())
        if result["input_signature"] != signature or result["scale"] != scale:
            raise ValueError(f"Changed input for cached nuclear signal: {key}")
    else:
        with np.load(image) as a:
            green, valid = a["green"], a["valid"]
        with np.load(mask) as a:
            labels = a["labels"]
        result = quantify_nuclear_mng(green, labels, valid, scale)
        result.update(well=well, day=int(day), scale=int(scale), input_signature=signature)
        assert result["mng_nuclei_count_check"] == count, key
        cache.write_text(json.dumps(result, indent=2))
    assert result["mng_nuclei_count_check"] == count, key
    return {k: v for k, v in result.items() if k != "input_signature"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = args.output.resolve()
    source = root / "well_day_metrics.csv"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    df = pd.read_csv(source)
    expected = {
        (f"{r}{c:02d}", d) for r in "CDEFGHIJKLMN" for c in range(3, 11) for d in range(5, 18)
    }
    assert len(df) == 1248 and set(zip(df.well, df.day)) == expected
    target = root / "nuclear_mng"
    (target / "frames").mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(root), r.well, int(r.day), int(r.nuclei_count), int(r.scale)) for r in df.itertuples()
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(measure, job) for job in jobs]):
            rows.append(future.result())
            if len(rows) % 48 == 0 or len(rows) == len(jobs):
                print(f"Nuclear mNG: {len(rows)}/{len(jobs)} frames", flush=True)
    result = pd.DataFrame(rows).sort_values(["well", "day"])
    assert set(zip(result.well, result.day)) == expected and len(result) == 1248
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    result.to_csv(root / "nuclear_mng_metrics.csv", index=False)
    (target / "provenance.json").write_text(
        json.dumps(
            {
                "completed": True,
                "frames": len(result),
                "created": datetime.now().astimezone().isoformat(),
                "source_measurements_sha256": source_hash,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "definition": "Equal-weight mean of per-nucleus raw mNG means; saved green camera units and existing labels; no background or illumination correction",
                "integrated_signal": "Sum of green within all nuclear masks multiplied by scale squared",
            },
            indent=2,
        )
    )
    print("Nuclear mNG measurement complete; original measurements unchanged.", flush=True)


if __name__ == "__main__":
    main()
