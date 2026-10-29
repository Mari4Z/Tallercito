"""Elige el umbral de alerta usando SOLO las puntuaciones de validación (el test no se toca).

Regla fijada antes de mirar el test: el umbral más bajo (más recall) cuya tasa de falsos positivos (FPR) en
validación no supere FPR_MAX. Uso: python src/select_threshold.py <ID_EXPERIMENTO>   (p. ej. E2_hgb)
"""
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from modeling import classification_metrics  # noqa: E402

FPR_MAX = 0.05
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def main(exp_id):
    row = pd.read_csv("evidencias/experimentos.csv").query("id == @exp_id").iloc[0]
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    path = mlflow.artifacts.download_artifacts(run_id=row.run_id, artifact_path="valid_scores.csv")
    v = pd.read_csv(path)
    curve = {float(t): classification_metrics(v.y_true, v.score_attack, t) for t in GRID}
    elegido = min(t for t, m in curve.items() if m["fpr"] <= FPR_MAX)
    pick = lambda m: {k: round(float(m[k]), 4) for k in ("recall", "precision", "fpr", "macro_f1", "tp", "fp", "tn", "fn")}
    out = {
        "experimento": exp_id, "run_id": row.run_id, "datos": "solo validación (data/prepared/valid.csv)",
        "regla": f"umbral mínimo con FPR <= {FPR_MAX} en validación (maximiza recall bajo un presupuesto de falsas alarmas)",
        "razon": "Un ataque ignorado es más costoso que una falsa alarma, pero un FPR alto satura al equipo de seguridad. "
                 "Se acepta como máximo 1 falsa alarma por cada 20 conexiones normales.",
        "umbral_elegido": elegido, "metricas_validacion_en_umbral_elegido": pick(curve[elegido]),
        "referencia_umbral_0.5": pick(curve[0.5]),
        "referencia_umbral_max_macro_f1": {"umbral": max(curve, key=lambda t: curve[t]["macro_f1"]),
                                           **pick(curve[max(curve, key=lambda t: curve[t]["macro_f1"])])},
        "curva_validacion": {str(t): pick(m) for t, m in curve.items() if round(t * 100) % 5 == 0},
        "nota": "El umbral se ajusta sobre la misma validación con la que se reportan sus métricas, por lo que estas son algo optimistas.",
    }
    Path("evidencias/umbral.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("umbral_elegido", "metricas_validacion_en_umbral_elegido", "referencia_umbral_0.5")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
