"""Prueba de recuperación de v1 y v2.

  python src/recover.py original    # entorno original: fija la muestra y guarda las salidas de referencia
  python src/recover.py recuperar   # entorno nuevo: verifica datos DVC, descarga v1/v2 por SHA y compara salidas

Muestra: 50 filas de data/prepared/valid.csv (semilla 42). Tolerancia numérica en probabilidades: 1e-9.
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd
import skops.io as sio

sys.path.insert(0, str(Path(__file__).parent))
from hf_model import load, predict, sha256  # noqa: E402

OUT = Path("evidencias/recuperacion")
N, SEED, TOL = 50, 42, 1e-9
HF = json.loads(Path("evidencias/huggingface.json").read_text())
AUD = json.loads(Path("evidencias/auditoria.json").read_text())


def sample_from_valid():
    v = pd.read_csv("data/prepared/valid.csv")
    return v.sample(n=N, random_state=SEED).sort_values("row_id").reset_index(drop=True)


def original():
    OUT.mkdir(parents=True, exist_ok=True)
    s = sample_from_valid()
    s.to_csv(OUT / "entrada.csv", index=False)
    for v in ("v1", "v2"):
        info = json.loads(next(Path("evidencias/mlflow").glob(f"*{HF['versiones'][v]['run_id']}/run.json")).read_text())
        path = next(Path(p) for p in glob.glob(f"mlruns/*/models/{info['model']['logged_model_id']}/artifacts/model.skops"))
        assert sha256(path) == HF["versiones"][v]["model_sha256"]
        model = sio.load(path, trusted=sio.get_untrusted_types(file=path))
        cfg = {"threshold": HF["versiones"][v]["umbral"],
              "input_features": [c for c in s.columns if c not in ("row_id", "group_id", "attack_cat", "label")]}
        out = predict(model, cfg, s)
        out.insert(0, "row_id", s.row_id)
        out.to_csv(OUT / f"salidas_originales_{v}.csv", index=False)
        print(v, "salidas originales guardadas:", len(out), "filas, umbral", cfg["threshold"])


def recuperar():
    res = {"datos": {}, "versiones": {}}
    # 1) datos recuperados con DVC: huellas contra las auditadas
    for k, a in AUD["archivos"].items():
        h = sha256(a["archivo"]); res["datos"][a["archivo"]] = {"sha256": h, "coincide_con_auditoria": h == a["sha256"]}
    for k, sp in AUD["particion"]["splits"].items():
        h = sha256(sp["archivo"]); res["datos"][sp["archivo"]] = {"sha256": h, "coincide_con_auditoria": h == sp["sha256_archivo"]}
    assert all(d["coincide_con_auditoria"] for d in res["datos"].values()), "Huellas de datos distintas a las auditadas"
    # 2) la muestra reconstruida desde los datos recuperados es la entregada
    s = sample_from_valid(); entrada = pd.read_csv(OUT / "entrada.csv")
    try:
        pd.testing.assert_frame_equal(s, entrada, check_dtype=False, atol=1e-12)
        misma = True
    except AssertionError as e:
        misma = False
        print("La muestra reconstruida difiere:", str(e)[:300])
    res["entrada"] = {"sha256": sha256(OUT / "entrada.csv"), "filas": len(entrada), "igual_a_muestra_reconstruida_desde_datos_DVC": misma}
    # 3) v1 y v2 descargados por SHA completo
    for v in ("v1", "v2"):
        sha = HF["versiones"][v]["commit_sha_hf"]
        model, cfg = load(HF["repo"], sha)
        out = predict(model, cfg, entrada); out.insert(0, "row_id", entrada.row_id)
        rec = OUT / f"salidas_recuperadas_{v}.csv"; out.to_csv(rec, index=False)
        ref = pd.read_csv(OUT / f"salidas_originales_{v}.csv")
        diff = float((out.score_attack.values - ref.score_attack.values).__abs__().max())
        res["versiones"][v] = {"hf_commit_sha": sha, "umbral": cfg["threshold"], "model_sha256_descargado": cfg["model_sha256"],
                               "filas": len(out), "max_dif_abs_score_attack": diff, "tolerancia": TOL,
                               "y_pred_identicos": bool((out.y_pred.values == ref.y_pred.values).all()),
                               "row_id_identicos": bool((out.row_id.values == ref.row_id.values).all()),
                               "coincide": diff <= TOL and bool((out.y_pred.values == ref.y_pred.values).all()),
                               "sha256_salidas_originales": sha256(OUT / f"salidas_originales_{v}.csv"), "sha256_salidas_recuperadas": sha256(rec),
                               "n_ataque_predichos": int(out.y_pred.sum())}
    res["resultado_global"] = "OK" if all(x["coincide"] for x in res["versiones"].values()) and res["entrada"]["igual_a_muestra_reconstruida_desde_datos_DVC"] else "FALLO"
    (OUT / "resultado.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    sys.exit(0 if res["resultado_global"] == "OK" else 1)


if __name__ == "__main__":
    {"original": original, "recuperar": recuperar}[sys.argv[1]]()
