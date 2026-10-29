# Entrega — MLOps · Defiende el servidor

## Integrantes y grupo
- **María José Cortés Romero** — Grupo A (entrega individual).

## Repositorio de código
- GitHub: https://github.com/Mari4Z/Tallercito
- Commit final de la entrega: el apuntado por el tag **`entrega`** (su SHA completo va en el mensaje de entrega; un commit no puede contener su propio SHA).
- Tags de código: `data-v1` (datos preparados), `modelo-v1` = `846fecafedb143276d6f5bf384ec3bb3279cb906`, `modelo-v2` = `6b50021dd5895a1c014d1f076a4ca9c60f8ba0ae`.

## Versiones del modelo
Repositorio de modelos: https://huggingface.co/Mari4Z/unsw-nb15-defiende-el-servidor

| | **v1** | **v2** |
|---|---|---|
| Modelo | Regresión logística (E1_logreg) | HistGradientBoosting (E2_hgb) |
| Umbral | 0.5 | 0.65 |
| Commit del código | `846fecafedb143276d6f5bf384ec3bb3279cb906` entrenó el modelo (tag `modelo-v1`: `846fecafedb143276d6f5bf384ec3bb3279cb906`) | `846fecafedb143276d6f5bf384ec3bb3279cb906` entrenó el modelo; umbral y selección congelados en `6b50021dd5895a1c014d1f076a4ca9c60f8ba0ae` (tag `modelo-v2`) |
| Datos (DVC, preparados) | `44585757006074b4f38735eed9dd9e08.dir` | `44585757006074b4f38735eed9dd9e08.dir` |
| Run ID de MLflow | `5892811f742340659a3b88199cd2ec23` | `09cf945f4cc14571a1e6ec7eb84c75b7` |
| Hugging Face, commit SHA | `308d2e786ae572a28a53dbe5910d6afd254d7445` | `53384879be28769b7ac007ea96ef1e53f9e48271` (revisión evaluada en test) |
| SHA-256 del modelo | `76d6c306f707cc952e5e82a1dbb5942c0fb3e75c2b28dd00a034fb91ea44ca52` | `544c94bdd84c124145588960ca25ad24f6dc4f32a137bfb0148fbf933dd711bd` |

Nota v2: después de evaluar solo se actualizó la model card (commit HF `246ca76bf2cf1df0fa0e6eb17170ebbb0e303ff7`); `model.skops` es idéntico al de la revisión evaluada.

Datos DVC: originales `fe93051ae209ffb0128a31e7a478c009.dir` · preparados `44585757006074b4f38735eed9dd9e08.dir`. Los CSV originales coinciden con los SHA-256 del anexo del taller.

## Dónde está cada evidencia
| Evidencia | Ruta |
|---|---|
| Auditoría de datos y particiones | `evidencias/auditoria.json` |
| Tabla de experimentos (4 configuraciones) | `evidencias/experimentos.csv` · hipótesis en `configs/experiments.yaml` |
| Registros de MLflow exportados | `evidencias/mlflow/` |
| Umbral elegido solo con validación | `evidencias/umbral.json` (commit `1667e8ded0e81c0415d377355ff29d74312f9263`) |
| Selección congelada antes del test | `evidencias/seleccion.json` |
| Test final: métricas y predicciones por fila | `evidencias/metricas-test.json`, `evidencias/predicciones-test.csv` |
| Comparación v1/v2 | `evidencias/comparacion-v1-v2.md` |
| Publicación en Hugging Face | `evidencias/huggingface.json` |
| Prueba de recuperación en entorno nuevo | `evidencias/recuperacion/` (entrada, salidas, hashes, `log-recuperacion.txt`, `resultado.json`) |
| Informe | `informe.pdf` (se envía adjunto) |

**Almacenamiento DVC:** dataset público de Hugging Face https://huggingface.co/datasets/Mari4Z/unsw-nb15-dvc-storage (lectura anónima; es el remoto por defecto).
Acceso: `dvc pull`, sin credenciales. Copia de respaldo con autenticación en DagsHub (`dagshub`), no necesaria para la revisión.

## Comandos (ver README para el detalle)
```bash
git clone https://github.com/Mari4Z/Tallercito.git && cd Tallercito          # instalación
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
dvc pull                                                                       # recuperación de datos
python src/run_experiments.py                                                  # entrenamiento (E0..E3, ~30 s)
python src/predict.py --version v2 --input evidencias/recuperacion/entrada.csv --output salida.csv   # predicción
bash scripts/prueba_recuperacion.sh /ruta/nueva                                # prueba completa en entorno nuevo
```

## Tiempo y recursos medidos
Equipo: Apple M4 (10 núcleos), 16 GB de RAM, macOS 15.6.1, Python 3.14.6, solo CPU.
- Preparación de datos (`prepare_data.py`): 9,7 s.
- Entrenamiento + evaluación + registro en MLflow: Dummy 2,6 s · regresión logística 5,4 s · HistGradientBoosting 13,6 s · random forest 5,9 s.
- Datos: 45 MB originales + 35 MB preparados (80 MB en el almacenamiento DVC). Modelos: v1 0,32 MB, v2 2,06 MB. Entorno virtual: ~900 MB.
- No medí el tiempo exacto de instalación de dependencias ni de `dvc pull`.

## Limitaciones conocidas
- Benchmark histórico generado en laboratorio; no prueba eficacia sobre una red actual.
- En test, v2 genera muchas más falsas alarmas que en validación (FPR 0.175 vs 0.049); la causa probable (otra mezcla de tráfico en el test oficial) es una hipótesis no comprobada.
- `sttl` y `ct_state_ttl` separan casi solas las clases por cómo se montó el laboratorio; quitarlas no baja el rendimiento porque otras variables llevan información equivalente.
- 42.38 % de las filas de entrenamiento tiene entradas repetidas y hay 229 grupos con etiquetas contradictorias; 10.37 % del test coincide con entradas de train/validación.
- Las métricas de validación de v2 son algo optimistas: el umbral se ajustó sobre esa misma validación.
- Solo clasificación binaria; las familias de ataque (p. ej. Worms, 130 filas) no se modelaron.
- La fecha de descarga de los CSV no se registró; consta la fecha de modificación del archivo local (`evidencias/auditoria.json`).

## Uso de IA, código externo y contribución
- **IA:** se usó **Claude Code** (asistente de Anthropic, modelo Claude Sonnet 5.5) para escribir el código (`src/`, `scripts/`), ejecutar los experimentos, exportar evidencias, redactar `EDA.md`, el README, este índice y el informe, y para operar Git, DVC, MLflow y Hugging Face bajo dirección de la autora.
- **Qué se verificó:** SHA-256 de los CSV contra el anexo; ausencia de grupos compartidos entre train y validación; recarga de los modelos desde su archivo con reproducción de las puntuaciones de validación (dif. máx. 1e-16); métricas de test recalculadas desde `predicciones-test.csv`; prueba de recuperación en entorno nuevo y sin credenciales; revisión de que el repositorio no contiene tokens ni datos.
- **Código externo:** bibliotecas con las versiones de `requirements.txt` (scikit-learn, pandas, numpy, DVC, MLflow, skops, huggingface_hub), según sus licencias. No se adaptó código de terceros.
- **Contribución:** trabajo individual de María José Cortés Romero (decisiones de diseño: modelos elegidos, regla del umbral y publicación de los datos), con la asistencia de IA descrita.
