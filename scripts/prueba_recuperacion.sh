#!/usr/bin/env bash
# Prueba de recuperación en carpeta y entorno nuevos (sin cachés de la ejecución original).
# Uso: bash prueba_recuperacion.sh <carpeta_nueva_vacía>
set -euxo pipefail
DEST="${1:?indica una carpeta nueva}"
mkdir -p "$DEST" && cd "$DEST"
export HF_HOME="$DEST/hf_cache"            # caché de Hugging Face nueva
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN      # sin credenciales: lectura pública
date -Iseconds
python3 --version
git clone https://github.com/Mari4Z/Tallercito.git proyecto   # 1) código
cd proyecto
git rev-parse HEAD
python3 -m venv .venv                      # 2) entorno nuevo y dependencias fijadas
.venv/bin/pip install -q -r requirements.txt
.venv/bin/dvc remote list                  # 3) datos originales y preparados desde el remoto (DagsHub), sin credenciales
.venv/bin/dvc pull
.venv/bin/python src/recover.py recuperar  # 4) verifica huellas, descarga v1/v2 por SHA completo, predice y compara
