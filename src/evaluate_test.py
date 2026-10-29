"""Evaluación FINAL en test de la solución congelada (evidencias/seleccion.json). Se ejecuta una sola vez.

El modelo se descarga de Hugging Face por su commit SHA completo; no se ajusta ni se cambia el umbral.
Uso: python src/evaluate_test.py   (se niega a repetirse si ya existe evidencias/metricas-test.json)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hf_model import load, predict, sha256  # noqa: E402
from modeling import classification_metrics  # noqa: E402

SEL = json.loads(Path("evidencias/seleccion.json").read_text())
OUT_M, OUT_P = Path("evidencias/metricas-test.json"), Path("evidencias/predicciones-test.csv")
TEST = Path("data/raw/UNSW_NB15_testing-set.csv")


def main():
    if OUT_M.exists():
        sys.exit("El test ya se evaluó (existe metricas-test.json). No se repite: la evaluación final es una sola.")
    aud = json.loads(Path("evidencias/auditoria.json").read_text())
    assert sha256(TEST) == aud["archivos"]["test"]["sha256"], "El archivo de test no coincide con el auditado"
    model, cfg = load(SEL["repositorio_modelos"], SEL["hf_commit_sha_completo"])
    assert cfg["model_sha256"] == SEL["hash_modelo_sha256"] and cfg["threshold"] == SEL["umbral"]
    test = pd.read_csv(TEST)
    assert len(test) == aud["archivos"]["test"]["filas"], "No se deben eliminar filas de test"
    pred = predict(model, cfg, test)
    out = pd.DataFrame({"row_id": "testing-set:" + test["id"].astype(str), "y_true": test["label"],
                        "score_attack": pred.score_attack.round(10), "y_pred": pred.y_pred})
    out.to_csv(OUT_P, index=False)
    m = classification_metrics(out.y_true, pred.score_attack, cfg["threshold"])
    OUT_M.write_text(json.dumps({
        "version": SEL["version_elegida"], "hf_repo": SEL["repositorio_modelos"], "hf_commit_sha": SEL["hf_commit_sha_completo"],
        "hash_modelo_sha256": sha256(Path(__import__("huggingface_hub").snapshot_download(SEL["repositorio_modelos"], revision=SEL["hf_commit_sha_completo"])) / "model.skops"),
        "hash_dataset_test_sha256": sha256(TEST), "filas_evaluadas": len(test), "umbral": cfg["threshold"],
        "regla_de_umbral": "y_pred = 1 si score_attack >= umbral", "clase_positiva": "1 = ataque",
        "matriz_de_confusion": {"tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]},
        "metricas": {k: m[k] for k in ("recall", "precision", "fpr", "f1_attack", "macro_f1", "accuracy", "zero_division")},
        "politica_denominador_cero": "se devuelve 0.0 y zero_division=true",
        "cruce_con_train_valid": "ver evidencias/auditoria.json (cruces_entre_particiones)"}, indent=2, ensure_ascii=False))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, indent=2))


if __name__ == "__main__":
    main()
