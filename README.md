# Defiende el servidor — UNSW-NB15 (MLOps, 2026-II)

Clasificador binario de conexiones de red (**0 = normal, 1 = ataque**, ataque = clase positiva) con trazabilidad completa:
código en **Git/GitHub**, datos en **DVC**, experimentos en **MLflow** y dos versiones del modelo en **Hugging Face Hub**.

- Código: https://github.com/Mari4Z/Tallercito
- Modelos (v1 y v2): https://huggingface.co/Mari4Z/unsw-nb15-defiende-el-servidor
- Almacenamiento DVC (lectura pública): https://huggingface.co/datasets/Mari4Z/unsw-nb15-dvc-storage
- Identificadores completos (SHA, hashes, run IDs): [entrega.md](entrega.md)

| Versión | Modelo | Umbral | Validación: recall / FPR / macro-F1 |
|---|---|---|---|
| v1 | Regresión logística | 0.50 | 0.975 / 0.133 / 0.930 |
| v2 | HistGradientBoosting | 0.65 | 0.953 / 0.049 / 0.946 |

Test final de v2 (una sola evaluación, 82.332 filas): recall 0.965, **FPR 0.175**, macro-F1 0.900. Empeora frente a validación; ver `EDA.md`
y las model cards. Es un benchmark histórico de laboratorio: no prueba eficacia sobre una red actual.

## Requisitos
Python **3.14.6** (probado en macOS arm64) y Git. Dependencias fijadas en `requirements.txt`.

## Instalación
```bash
git clone https://github.com/Mari4Z/Tallercito.git && cd Tallercito
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Recuperar los datos (DVC)
```bash
dvc pull          # remoto por defecto: dataset público de Hugging Face, solo lectura, sin credenciales
```
Recupera `data/raw/` (los dos CSV originales) y `data/prepared/` (`train.csv`, `valid.csv`). Las huellas SHA-256 esperadas están en
`evidencias/auditoria.json` (los originales coinciden con el anexo del taller). `dvc push` solo funciona con el remoto `dagshub`
(requiere credenciales del autor, guardadas fuera del repositorio).

## Reproducir los experimentos
```bash
python src/prepare_data.py     # audita, verifica SHA-256 y recrea train/valid (resultado idéntico: semilla 42, grupos de entradas idénticas)
python src/run_experiments.py  # entrena E0..E3 (configs/experiments.yaml) y los registra en MLflow local (mlflow.db); ~30 s
python src/export_mlflow.py    # exporta evidencias/experimentos.csv y evidencias/mlflow/
mlflow ui --backend-store-uri sqlite:///mlflow.db      # opcional: explorar los runs
```
`run_experiments.py` exige que `src/`, `configs/` y `data/prepared.dvc` estén commiteados (así cada run queda ligado a un commit).
Cada ejecución crea runs nuevos en MLflow.

## Descargar un modelo y predecir
Cada versión se descarga por su **commit SHA completo** de Hugging Face (los SHA están en `evidencias/huggingface.json`):
```bash
python src/predict.py --version v1 --input evidencias/recuperacion/entrada.csv --output salida_v1.csv
python src/predict.py --version v2 --input evidencias/recuperacion/entrada.csv --output salida_v2.csv
```
Salida: `row_id, score_attack, y_pred`, con `y_pred = 1` si `score_attack >= umbral` (0.5 en v1, 0.65 en v2). `hf_model.load` verifica el
SHA-256 del modelo contra `config.json`. En Python:
```python
import sys; sys.path.insert(0, "src")
from hf_model import load, predict
model, cfg = load("Mari4Z/unsw-nb15-defiende-el-servidor", "53384879be28769b7ac007ea96ef1e53f9e48271")   # v2
```

## Prueba de recuperación en entorno nuevo
```bash
bash scripts/prueba_recuperacion.sh /ruta/a/carpeta/nueva
```
Clona el código, crea un entorno, hace `dvc pull`, descarga v1 y v2 por SHA y compara las predicciones sobre 50 filas fijas de validación con las
originales (tolerancia 1e-9). Resultado y log en `evidencias/recuperacion/`.

## Métricas
Clase positiva = ataque. `recall = TP/(TP+FN)`, `precision = TP/(TP+FP)`, `FPR = FP/(FP+TN)`, `macro-F1` = promedio del F1 de la clase ataque y
el de la clase normal. **Denominador cero:** el cociente vale 0.0 y se marca `zero_division = true` (`src/modeling.py`); no ocurrió en ningún run.
La accuracy se registra como complemento, no para decidir.

## Mapa del repositorio
| Ruta | Contenido |
|---|---|
| `EDA.md`, `src/eda.py` | Análisis exploratorio y hallazgos (fuga por `id`, duplicados, `sttl`, etc.) |
| `configs/` | `data.yaml` (semilla, particiones), `experiments.yaml` (hipótesis y configuraciones), `conclusiones.yaml` |
| `src/` | `prepare_data.py`, `modeling.py`, `run_experiments.py`, `export_mlflow.py`, `select_threshold.py`, `publish_hf.py`, `hf_model.py`, `predict.py`, `evaluate_test.py`, `recover.py` |
| `data/*.dvc` | Punteros DVC: `raw.dvc` (datos recibidos) y `prepared.dvc` (datos preparados y particiones) |
| `evidencias/` | `auditoria.json`, `experimentos.csv`, `seleccion.json`, `umbral.json`, `metricas-test.json`, `predicciones-test.csv`, `mlflow/`, `recuperacion/`, `huggingface.json`, `comparacion-v1-v2.md` |

## Atribución
Moustafa, N. y Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems.* MilCIS, IEEE.
Fuente oficial: https://research.unsw.edu.au/projects/unsw-nb15-dataset (uso académico con cita).
