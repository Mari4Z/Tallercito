"""Auditoría de UNSW-NB15 y creación de particiones train/validación.

Uso (desde la raíz del repositorio):  python src/prepare_data.py
Genera data/prepared/{train,valid}.csv y evidencias/auditoria.json.
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

CFG = yaml.safe_load(Path("configs/data.yaml").read_text())
RAW, OUT = Path(CFG["raw_dir"]), Path(CFG["prepared_dir"])
TARGET, EXCLUDED = CFG["target"], CFG["excluded_from_inputs"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_of_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def file_audit(name):
    path = RAW / name
    df = pd.read_csv(path)
    digest = sha256(path)
    return df, {
        "archivo": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "sha256_esperado": CFG["expected_sha256"][name],
        "sha256_coincide": digest == CFG["expected_sha256"][name],
        "fecha_modificacion_archivo_local": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "filas": len(df),
        "filas_esperadas": CFG["expected_rows"][name],
        "columnas": df.shape[1],
        "nombres_columnas": list(df.columns),
        "tipos": {k: str(v) for k, v in df.dtypes.items()},
        "faltantes_total": int(df.isna().sum().sum()),
        "faltantes_por_columna": {k: int(v) for k, v in df.isna().sum().items() if v},
        "clases_label": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "clases_attack_cat": {k: int(v) for k, v in df["attack_cat"].value_counts().items()},
        "id_unico": bool(df["id"].is_unique),
    }


def group_stats(df, feats):
    """Duplicados por variables predictoras y grupos con etiquetas contradictorias."""
    gid = df.groupby(feats, sort=False, dropna=False).ngroup()
    sizes = gid.map(gid.value_counts())
    labels_per_group = df.groupby(gid)[TARGET].nunique()
    contradict = labels_per_group[labels_per_group > 1].index
    in_contra = gid.isin(contradict)
    return gid, {
        "grupos_unicos": int(gid.nunique()),
        "filas_duplicadas_extra": int(df.duplicated(subset=feats).sum()),
        "pct_filas_duplicadas_extra": round(float(df.duplicated(subset=feats).mean() * 100), 2),
        "filas_en_grupos_con_repeticion": int((sizes > 1).sum()),
        "grupos_contradictorios": int(len(contradict)),
        "filas_en_grupos_contradictorios": int(in_contra.sum()),
        "pct_filas_en_grupos_contradictorios": round(float(in_contra.mean() * 100), 3),
    }


def main():
    train, a_train = file_audit(CFG["train_file"])
    test, a_test = file_audit(CFG["test_file"])
    problemas = [n for n, a in (("train", a_train), ("test", a_test))
                 if not (a["sha256_coincide"] and a["filas"] == a["filas_esperadas"])]
    if problemas:
        sys.exit(f"Archivo(s) distinto(s) al esperado: {problemas}. Revisa la descarga antes de continuar.")
    if list(train.columns) != list(test.columns):
        sys.exit("Train y test no tienen las mismas columnas.")

    feats = [c for c in train.columns if c not in EXCLUDED]
    gid_train, dup_train = group_stats(train, feats)
    _, dup_test = group_stats(test, feats)

    # ---- partición: grupos de entradas idénticas nunca se reparten entre train y valid ----
    sgkf = StratifiedGroupKFold(n_splits=round(1 / CFG["val_fraction"]), shuffle=True, random_state=CFG["seed"])
    tr_idx, va_idx = next(sgkf.split(train[feats], train[TARGET], groups=gid_train))
    train["row_id"] = "training-set:" + train["id"].astype(str)
    train["group_id"] = gid_train
    parts = {"train": train.iloc[tr_idx], "valid": train.iloc[va_idx]}
    shared = set(parts["train"].group_id) & set(parts["valid"].group_id)
    assert not shared, "Hay grupos repartidos entre train y valid"
    assert not set(parts["train"].row_id) & set(parts["valid"].row_id)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["row_id", "group_id"] + feats + ["attack_cat", TARGET]
    splits = {}
    for name, p in parts.items():
        path = OUT / f"{name}.csv"
        p[cols].to_csv(path, index=False)
        splits[name] = {
            "archivo": str(path), "sha256_archivo": sha256(path), "filas": len(p),
            "grupos": int(p.group_id.nunique()),
            "clases_label": {str(k): int(v) for k, v in p[TARGET].value_counts().sort_index().items()},
            "pct_ataque": round(float(p[TARGET].mean() * 100), 2),
            "sha256_row_ids": sha_of_ids(p.row_id),
        }

    # ---- cruces entre particiones (el test solo se audita, sus etiquetas no se usan) ----
    key = lambda d: pd.util.hash_pandas_object(d[feats], index=False)
    k_tr, k_va, k_te = key(parts["train"]), key(parts["valid"]), key(test)
    in_tr, in_va = k_te.isin(set(k_tr)), k_te.isin(set(k_va))
    cruces = {
        "filas_test_con_entrada_igual_en_train_split": int(in_tr.sum()),
        "filas_test_con_entrada_igual_en_valid_split": int(in_va.sum()),
        "filas_test_con_entrada_igual_en_train_o_valid": int((in_tr | in_va).sum()),
        "pct_test_solapado_con_train_o_valid": round(float((in_tr | in_va).mean() * 100), 2),
        "filas_valid_con_entrada_igual_en_train_split": int(k_va.isin(set(k_tr)).sum()),
    }

    audit = {
        "generado": dt.datetime.now().isoformat(timespec="seconds"),
        "fuente": {
            "referencia_oficial_y_atribucion": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
            "origen_de_la_copia": "copia espejo de terceros enlazada en el anexo del taller; los SHA-256 coinciden con los publicados allí (no autentican contra el archivo oficial)",
            "fecha_de_descarga": "pendiente de confirmar por el equipo; la fecha de modificación del archivo local está en cada archivo",
        },
        "archivos": {"train": a_train, "test": a_test},
        "variables": {
            "objetivo": TARGET, "clase_positiva": "1 = ataque",
            "excluidas_de_las_entradas": EXCLUDED,
            "usadas": feats, "n_usadas": len(feats),
            "tipos_usados": {"categoricas": [c for c in feats if train[c].dtype == object or str(train[c].dtype) == "str"],
                             "numericas": [c for c in feats if not (train[c].dtype == object or str(train[c].dtype) == "str")]},
        },
        "duplicados_por_variables_predictoras": {"train": dup_train, "test": dup_test},
        "politica_de_limpieza": {
            "duplicados": "No se eliminan filas. Cada grupo de entradas idénticas (mismas 42 variables predictoras) se asigna completo a train o a valid, de modo que ninguna entrada idéntica queda en ambos lados. Conserva la distribución original y el peso de las conexiones repetidas.",
            "grupos_contradictorios": "Se conservan y se asignan completos a un solo lado. Son ruido de etiquetas que fija un techo al rendimiento; se reportan aquí y en el informe.",
            "test": "No se elimina ninguna fila. Se evalúa una sola vez la partición completa.",
            "imputacion_y_escalado": "No se aplican en la preparación: no hay faltantes; codificación y escalado se ajustan dentro del pipeline sólo con train.",
        },
        "particion": {
            "metodo": "StratifiedGroupKFold(n_splits=5, shuffle=True), primer fold como validación (~20 %)",
            "grupo": "filas con las mismas 42 variables predictoras",
            "semilla": CFG["seed"],
            "grupos_compartidos_train_valid": len(shared),
            "splits": splits,
        },
        "cruces_entre_particiones": cruces,
    }
    Path("evidencias").mkdir(exist_ok=True)
    Path("evidencias/auditoria.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps({"splits": splits, "cruces": cruces, "dup_train": dup_train, "dup_test": dup_test}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
