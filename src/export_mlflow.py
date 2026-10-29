"""Exporta los runs de MLflow a evidencias/ sin depender del servidor/base de datos local.

Genera evidencias/experimentos.csv y evidencias/mlflow/<id>_<run_id>/{run.json,model_metadata/}.
Uso: python src/export_mlflow.py
"""
import hashlib
import json
import shutil
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
exp = client.get_experiment_by_name(yaml.safe_load(Path("configs/experiments.yaml").read_text())["mlflow_experiment"])
models = {m.source_run_id: m for m in mlflow.search_logged_models(experiment_ids=[exp.experiment_id], output_format="list")}
CONCL = yaml.safe_load(Path("configs/conclusiones.yaml").read_text())
OUT = Path("evidencias/mlflow")
shutil.rmtree(OUT, ignore_errors=True)
rows = []
for r in sorted(client.search_runs([exp.experiment_id], filter_string="attributes.status = 'FINISHED'"), key=lambda r: r.info.start_time):
    eid = r.data.tags["experiment_id"]
    d = OUT / f"{eid}_{r.info.run_id}"
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "run_id": r.info.run_id, "run_name": r.info.run_name, "status": r.info.status,
        "start_time_ms": r.info.start_time, "artifact_uri": r.info.artifact_uri,
        "params": r.data.params, "metrics": r.data.metrics,
        "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.log-model")}}, indent=2, ensure_ascii=False))
    # metadatos del modelo (sin los pesos, que se publican en Hugging Face Hub) + hash del artefacto
    lm = models.get(r.info.run_id)
    if lm:
        src = Path(lm.artifact_location.replace("file://", ""))
        (d / "model_metadata").mkdir()
        for f in src.iterdir():
            if f.is_file() and not f.name.startswith("model."):
                shutil.copy(f, d / "model_metadata" / f.name)
        weights = next(src.glob("model.*"))
        info = json.loads((d / "run.json").read_text())
        info["model"] = {"logged_model_id": lm.model_id, "archivo": weights.name, "bytes": weights.stat().st_size,
                         "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
        (d / "run.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    p, m, t = r.data.params, r.data.metrics, r.data.tags
    rows.append({
        "id": eid, "hipotesis": t["hypothesis"], "cambio": t["change"], "familia": t["family"],
        "parametros": json.dumps({k: v for k, v in p.items() if k.startswith("clf_") or k in ("dropped_features", "n_features")}, ensure_ascii=False),
        "semilla": p["seed"], "semilla_particion": p["split_seed"],
        "commit_git": t["git_commit"], "referencia_dvc_datos_preparados": t["dvc_prepared_md5"], "tag_datos": t["git_tag_datos"],
        "run_id": r.info.run_id, "sha256_modelo": json.loads((d / "run.json").read_text()).get("model", {}).get("sha256", ""), "umbral": m["threshold"],
        "recall": round(m["recall"], 4), "precision": round(m["precision"], 4), "fpr": round(m["fpr"], 4),
        "macro_f1": round(m["macro_f1"], 4), "accuracy": round(m["accuracy"], 4),
        "tp": int(m["tp"]), "fp": int(m["fp"]), "tn": int(m["tn"]), "fn": int(m["fn"]),
        "conclusion_validacion": " ".join(CONCL[eid].split()),
    })
pd.DataFrame(rows).to_csv("evidencias/experimentos.csv", index=False)
print(pd.DataFrame(rows)[["id", "familia", "recall", "precision", "fpr", "macro_f1", "run_id"]].to_string(index=False))
