"""Predice con una versión publicada (v1 o v2) descargándola de Hugging Face Hub por su commit SHA completo.

Uso: python src/predict.py --version v2 --input entradas.csv --output salida.csv
El CSV de entrada debe tener las 42 columnas de entrada (ver config.json de la revisión); si trae row_id se conserva.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from hf_model import load, predict

ap = argparse.ArgumentParser()
ap.add_argument("--version", choices=["v1", "v2"], required=True)
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()
hf = json.loads((Path(__file__).parent.parent / "evidencias/huggingface.json").read_text())
sha = hf["versiones"][a.version]["commit_sha_hf"]
model, cfg = load(hf["repo"], sha)
df = pd.read_csv(a.input)
out = predict(model, cfg, df)
if "row_id" in df:
    out.insert(0, "row_id", df["row_id"])
out.to_csv(a.output, index=False)
print(f"{a.version} (HF {sha[:10]}, umbral {cfg['threshold']}): {len(out)} filas -> {a.output}; ataques predichos: {int(out.y_pred.sum())}")
