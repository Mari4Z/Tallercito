"""Entrena las configuraciones de configs/experiments.yaml y las registra en MLflow.

Uso: python src/run_experiments.py   (desde la raíz; requiere data/prepared/ y un árbol Git limpio)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from modeling import CATEGORICAL, build_pipeline, classification_metrics  # noqa: E402

DATA = yaml.safe_load(Path("configs/data.yaml").read_text())
EXP = yaml.safe_load(Path("configs/experiments.yaml").read_text())
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "unsw-nb15-defiende-el-servidor"
NON_FEATURES = ["row_id", "group_id", "attack_cat", DATA["target"]]


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def dvc_md5(dvc_file):
    return next(l.split(":", 1)[1].strip() for l in Path(dvc_file).read_text().splitlines() if "md5:" in l)


def main():
    dirty = bool(git("status", "--porcelain", "--", "src", "configs", "data/prepared.dvc"))
    if dirty:
        sys.exit("Hay cambios sin commitear en src/, configs/ o data/prepared.dvc: haz commit para que el run quede ligado a un commit.")
    commit = git("rev-parse", "HEAD")
    data_ref = {"dvc_prepared_md5": dvc_md5("data/prepared.dvc"), "dvc_raw_md5": dvc_md5("data/raw.dvc"), "git_tag_datos": "data-v1"}

    train = pd.read_csv(Path(DATA["prepared_dir"]) / "train.csv")
    valid = pd.read_csv(Path(DATA["prepared_dir"]) / "valid.csv")
    feats = [c for c in train.columns if c not in NON_FEATURES]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    for e in EXP["experiments"]:
        used = [c for c in feats if c not in e["drop_features"]]
        numeric = [c for c in used if c not in CATEGORICAL]
        pipe = build_pipeline(e["model"]["type"], e["model"]["params"], numeric, EXP["seed"])
        with mlflow.start_run(run_name=e["id"]) as run:
            pipe.fit(train[used], train[DATA["target"]])
            score = pipe.predict_proba(valid[used])[:, 1]
            m = classification_metrics(valid[DATA["target"]], score, EXP["threshold"])

            mlflow.set_tags({"experiment_id": e["id"], "family": e["family"], "change": e["change"],
                             "hypothesis": " ".join(e["hypothesis"].split()), "git_commit": commit, **data_ref})
            mlflow.log_params({"model": e["model"]["type"], **{f"clf_{k}": v for k, v in e["model"]["params"].items()},
                               "seed": EXP["seed"], "split_seed": DATA["seed"], "n_features": len(used),
                               "dropped_features": ",".join(e["drop_features"]) or "none",
                               "train_rows": len(train), "valid_rows": len(valid)})
            mlflow.log_metrics({k: float(v) for k, v in m.items()})
            mlflow.sklearn.log_model(pipe, name="model")
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "valid_scores.csv"
                pd.DataFrame({"row_id": valid.row_id, "y_true": valid[DATA["target"]], "score_attack": score}).to_csv(p, index=False)
                mlflow.log_artifact(str(p))
            print(f"{e['id']:<16} run={run.info.run_id} recall={m['recall']:.4f} prec={m['precision']:.4f} "
                  f"fpr={m['fpr']:.4f} macroF1={m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
