"""Pipeline (preprocesamiento + estimador) y métricas de clasificación binaria."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL = ["proto", "service", "state"]
MODELS = {"LogisticRegression": LogisticRegression, "RandomForestClassifier": RandomForestClassifier}


def build_pipeline(model_type, params, numeric, seed):
    """Todo lo que se aprende (categorías, medias, desviaciones) se ajusta solo con train al llamar fit()."""
    cat = OneHotEncoder(handle_unknown="ignore")
    if model_type == "LogisticRegression":
        # colas pesadas -> log1p; escalado necesario para el modelo lineal
        num = Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())])
    else:
        num = "passthrough"  # los árboles no necesitan transformar las numéricas
    pre = ColumnTransformer([("cat", cat, CATEGORICAL), ("num", num, numeric)])
    return Pipeline([("pre", pre), ("clf", MODELS[model_type](random_state=seed, **params))])


def safe_div(a, b):
    """Política de denominador cero: devuelve 0.0 y se marca con zero_division=True."""
    return (a / b, False) if b else (0.0, True)


def classification_metrics(y_true, score, threshold):
    """Ataque (1) = clase positiva. y_pred = 1 si score >= umbral."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall, z1 = safe_div(tp, tp + fn)
    precision, z2 = safe_div(tp, tp + fp)
    fpr, z3 = safe_div(fp, fp + tn)
    f1_att, _ = safe_div(2 * precision * recall, precision + recall)
    prec_n, _ = safe_div(tn, tn + fn)      # precision de la clase normal
    rec_n, _ = safe_div(tn, tn + fp)       # recall de la clase normal
    f1_norm, _ = safe_div(2 * prec_n * rec_n, prec_n + rec_n)
    return {
        "threshold": threshold, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "recall": recall, "precision": precision, "fpr": fpr,
        "f1_attack": f1_att, "macro_f1": (f1_att + f1_norm) / 2,
        "accuracy": (tp + tn) / len(y_true), "zero_division": bool(z1 or z2 or z3),
    }
