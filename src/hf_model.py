"""Descarga una revisión (SHA completo) del modelo en Hugging Face Hub, verifica su hash y predice."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import skops.io as sio
from huggingface_hub import snapshot_download


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(repo_id, revision):
    """Carga preprocesamiento + modelo + umbral de una revisión. `revision` debe ser el commit SHA completo."""
    if len(revision) != 40:
        raise ValueError("Usa el commit SHA completo (40 caracteres), no una rama ni un tag.")
    folder = Path(snapshot_download(repo_id=repo_id, revision=revision))
    cfg = json.loads((folder / "config.json").read_text())
    weights = folder / "model.skops"
    if sha256(weights) != cfg["model_sha256"]:
        raise RuntimeError("El hash de model.skops no coincide con config.json.")
    # skops exige declarar los tipos no triviales; son los de scikit-learn/numpy del propio pipeline
    model = sio.load(weights, trusted=sio.get_untrusted_types(file=weights))
    return model, cfg


def predict(model, cfg, df):
    """Devuelve score_attack (probabilidad de ataque) y y_pred = 1 si score_attack >= umbral."""
    score = model.predict_proba(df[cfg["input_features"]])[:, 1]
    return pd.DataFrame({"score_attack": score, "y_pred": (score >= cfg["threshold"]).astype(int)}, index=df.index)
