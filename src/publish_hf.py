"""Publica v1 y v2 en un repositorio de modelos de Hugging Face Hub (dos commits, cada uno con su tag).

Uso: python src/publish_hf.py   (requiere `hf auth login`). Escribe evidencias/huggingface.json.
"""
import glob
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).parent))
from hf_model import sha256  # noqa: E402

REPO = "Mari4Z/unsw-nb15-defiende-el-servidor"
EXPS = {r["id"]: r for r in pd.read_csv("evidencias/experimentos.csv").to_dict("records")}
RUN_JSON = {json.loads((p / "run.json").read_text())["tags"]["experiment_id"]: p for p in Path("evidencias/mlflow").iterdir()}
UMBRAL = json.loads(Path("evidencias/umbral.json").read_text())
FEATURES = [c for c in pd.read_csv("data/prepared/valid.csv", nrows=1).columns if c not in ("row_id", "group_id", "attack_cat", "label")]
REQ = ["numpy==2.5.3", "pandas==3.0.6", "scikit-learn==1.9.1", "scipy==1.18.1", "joblib==1.6.0", "skops==0.16.0", "huggingface_hub==2.4.0"]

VERSIONS = {
    "v1": {"exp": "E1_logreg", "threshold": 0.5, "rule": "umbral por defecto 0.5 (referencia sencilla)",
           "desc": "Regresión logística (C=1.0) con log1p + estandarización de variables numéricas y one-hot de proto/service/state."},
    "v2": {"exp": "E2_hgb", "threshold": UMBRAL["umbral_elegido"], "rule": UMBRAL["regla"],
           "desc": "HistGradientBoosting (max_iter=200, parada temprana interna solo con train) con one-hot de proto/service/state."},
}


def model_file(exp_id):
    info = json.loads((RUN_JSON[exp_id] / "run.json").read_text())
    path = next(Path(p) for p in glob.glob(f"mlruns/*/models/{info['model']['logged_model_id']}/artifacts/model.skops"))
    assert sha256(path) == info["model"]["sha256"], "hash distinto al exportado de MLflow"
    return path, info


def validation_metrics(v, exp):
    if v == "v2":
        m = UMBRAL["metricas_validacion_en_umbral_elegido"]
        return {k: m[k] for k in ("recall", "precision", "fpr", "macro_f1", "tp", "fp", "tn", "fn")}
    return {k: exp[k] for k in ("recall", "precision", "fpr", "macro_f1", "tp", "fp", "tn", "fn")}


def preflight(path, info, run_id):
    """El archivo debe reproducir las puntuaciones de validación que MLflow guardó."""
    model = sio.load(path, trusted=sio.get_untrusted_types(file=path))
    valid = pd.read_csv("data/prepared/valid.csv").head(200)
    saved = pd.read_csv(glob.glob(f"mlruns/*/{run_id}/artifacts/valid_scores.csv")[0]).head(200)
    got = model.predict_proba(valid[FEATURES])[:, 1]
    diff = float(np.abs(got - saved.score_attack.to_numpy()).max())
    assert diff < 1e-9, f"las puntuaciones no coinciden (dif. máx {diff})"
    return diff


def card(v, spec, exp, info, m):
    extra = ("**Resultado en test:** no evaluado al publicar esta revisión. Se evalúa una sola vez después de congelar la selección "
             "(ver el repositorio de código)." if v == "v2" else "**Test:** esta versión no se evalúa en test (la comparación v1/v2 se hace con validación).")
    return f"""---
library_name: sklearn
tags: [tabular-classification, network-intrusion-detection, unsw-nb15, skops]
datasets: [UNSW-NB15]
metrics: [recall, precision, f1]
---
# Detección de ataques en UNSW-NB15 — {v}

Clasificador binario de conexiones de red: **0 = normal, 1 = ataque** (clase positiva). Proyecto académico de MLOps
(Universidad Externado de Colombia, Defiende el servidor). **No demuestra eficacia sobre una red real actual.**

## Qué es esta versión
{spec['desc']}

- Umbral de alerta: **{spec['threshold']}** — {spec['rule']}.
- Regla de decisión: `y_pred = 1` si `score_attack >= umbral`, con `score_attack = predict_proba(X)[:, 1]`.
- Entradas: 42 variables (ver `config.json`). Excluidas: `id`, `label`, `attack_cat`.

## Datos
UNSW-NB15, particiones publicadas `UNSW_NB15_training-set.csv` (175.341 filas) y `UNSW_NB15_testing-set.csv` (82.332).
Entrenamiento/validación se crean solo a partir del archivo de entrenamiento (140.272 / 35.069 filas) agrupando entradas idénticas
para que ningún grupo quede en ambos lados (semilla 42). Versión de datos DVC: `{exp['referencia_dvc_datos_preparados']}` (etiqueta Git `data-v1`).

## Métricas de validación (clase positiva = ataque, umbral {spec['threshold']})
| Recall | Precision | FPR | Macro-F1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|
| {m['recall']:.4f} | {m['precision']:.4f} | {m['fpr']:.4f} | {m['macro_f1']:.4f} | {int(m['tp'])} | {int(m['fp'])} | {int(m['tn'])} | {int(m['fn'])} |

{extra}

## Limitaciones
- Benchmark histórico generado en laboratorio; no prueba eficacia en una red actual.
- `sttl` y `ct_state_ttl` separan casi solas las clases por cómo se montó el laboratorio; quitarlas no baja el rendimiento porque otras variables llevan la misma información.
- El 42 % de las filas de entrenamiento tiene entradas repetidas y hay 229 grupos con etiquetas contradictorias (ruido que limita el rendimiento).
- El 10,4 % del test tiene entradas idénticas a train/validación, lo que puede inflar el resultado.
- El umbral se ajustó sobre la misma validación con la que se reportan sus métricas, que por tanto son algo optimistas.
- Las familias de ataque raras (p. ej. Worms, 130 filas) tienen muy pocos ejemplos.

## Uso
```python
from huggingface_hub import snapshot_download
import pandas as pd, sys
# desde el repositorio de código: src/hf_model.py
from hf_model import load, predict
model, cfg = load("{REPO}", revision="<COMMIT_SHA_COMPLETO_DE_ESTA_REVISION>")
print(predict(model, cfg, pd.read_csv("entradas.csv")))   # score_attack, y_pred
```
Instala las dependencias de `requirements.txt` (Python 3.14.6). El archivo `model.skops` se carga con `skops.io.load`;
`hf_model.load` verifica su SHA-256 contra `config.json`.

## Trazabilidad
| Pieza | Identificador |
|---|---|
| Código (commit Git que entrenó el modelo) | `{exp['commit_git']}` |
| Datos preparados (DVC) | `{exp['referencia_dvc_datos_preparados']}` |
| Experimento MLflow (run ID) | `{exp['run_id']}` |
| Modelo (SHA-256 de `model.skops`) | `{info['model']['sha256']}` |

## Atribución
Moustafa, N. y Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems.* MilCIS, IEEE.
Fuente oficial: https://research.unsw.edu.au/projects/unsw-nb15-dataset
"""


def main():
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    out = {"repo": REPO, "url": f"https://huggingface.co/{REPO}", "versiones": {}}
    for v, spec in VERSIONS.items():
        exp = EXPS[spec["exp"]]
        path, info = model_file(spec["exp"])
        diff = preflight(path, info, exp["run_id"])
        m = validation_metrics(v, exp)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            shutil.copy(path, d / "model.skops")
            (d / "config.json").write_text(json.dumps({
                "version": v, "model": spec["desc"], "positive_class": 1, "threshold": spec["threshold"],
                "threshold_rule": spec["rule"], "score": "predict_proba(X)[:, 1]", "decision": "y_pred = 1 if score_attack >= threshold",
                "input_features": FEATURES, "excluded_from_inputs": ["id", "label", "attack_cat"],
                "model_sha256": info["model"]["sha256"], "mlflow_run_id": exp["run_id"], "git_commit": exp["commit_git"],
                "dvc_prepared_md5": exp["referencia_dvc_datos_preparados"], "git_tag_data": "data-v1",
                "validation_metrics": m, "python": "3.14.6", "format": "skops"}, indent=2, ensure_ascii=False))
            (d / "requirements.txt").write_text("\n".join(REQ) + "\n")
            (d / "README.md").write_text(card(v, spec, exp, info, m))
            ci = api.upload_folder(repo_id=REPO, folder_path=str(d), repo_type="model",
                                   commit_message=f"{v}: {spec['exp']} (umbral {spec['threshold']})")
        api.create_tag(REPO, tag=v, revision=ci.oid, repo_type="model", tag_message=f"Versión {v}")
        out["versiones"][v] = {"experimento": spec["exp"], "run_id": exp["run_id"], "umbral": spec["threshold"],
                               "commit_sha_hf": ci.oid, "tag": v, "model_sha256": info["model"]["sha256"],
                               "git_commit_codigo": exp["commit_git"], "dvc_prepared_md5": exp["referencia_dvc_datos_preparados"],
                               "preflight_max_diff_scores": diff}
        print(v, ci.oid, "preflight dif máx:", diff)
    Path("evidencias/huggingface.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
