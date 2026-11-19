"""Regenerate report artifacts: python evaluate.py."""

import hashlib
import importlib.metadata
import json
import platform

from src.pipeline import DATA_PATH, ROOT, run_experiment


def main():
    result = run_experiment()
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    for key in ["validation_metrics", "test_metrics", "predictions"]:
        result[key].to_csv(output / f"{key}.csv", index=False)
    summary = {
        "selected_model": result["selected"], "selection_metric": "validation MAE",
        "audit": result["audit"],
        "splits": {key: {"rows": len(result[key]), "start": str(result[key].Date.min().date()),
                          "end": str(result[key].Date.max().date())} for key in ["train", "validation", "test"]},
        "test_metrics": result["test_metrics"].to_dict(orient="records"),
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "scikit-learn", "streamlit", "plotly"]},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
