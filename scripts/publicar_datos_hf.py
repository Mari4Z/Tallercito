"""Publica el almacenamiento de DVC (carpeta files/md5/...) como dataset público de Hugging Face Hub.

Uso: python scripts/publicar_datos_hf.py <carpeta_con_files/>   (requiere `hf auth login`)
La carpeta se genera con: dvc remote add --local tmp <ruta> && dvc push -r tmp
Se lee desde DVC como remoto HTTP de solo lectura (sin credenciales).
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Mari4Z/unsw-nb15-dvc-storage"
CARD = """---
license: other
pretty_name: UNSW-NB15 (almacenamiento DVC para un taller de MLOps)
tags: [dvc, network-intrusion-detection, unsw-nb15]
---
# Almacenamiento DVC de UNSW-NB15 (solo para reproducir un taller académico)

Este repositorio **no es el dataset oficial**: contiene los objetos de DVC (`files/md5/...`) del proyecto
https://github.com/Mari4Z/Tallercito, para poder recuperar con `dvc pull` los datos originales y preparados sin credenciales.

- Datos originales: copia de las particiones publicadas `UNSW_NB15_training-set.csv` y `UNSW_NB15_testing-set.csv`
  (SHA-256 en el anexo del taller y en `evidencias/auditoria.json` del repositorio de código).
- Datos preparados: `train.csv` y `valid.csv` derivados del archivo de entrenamiento (ver `src/prepare_data.py`).
- Es un benchmark histórico generado en laboratorio; no prueba eficacia sobre una red actual.

## Atribución y condiciones de uso
Moustafa, N. y Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems.* MilCIS, IEEE.
Fuente oficial, origen y condiciones de uso: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Úsalo solo con fines académicos y cita a los autores. Si los titulares piden retirarlo, se retira.

## Uso
```
dvc pull        # en el repositorio de código; el remoto por defecto apunta aquí
```
"""
src = Path(sys.argv[1])
api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=REPO, repo_type="dataset", commit_message="Ficha del dataset")
ci = api.upload_folder(repo_id=REPO, folder_path=str(src), repo_type="dataset", commit_message="Objetos DVC (datos originales y preparados)")
print("commit:", ci.oid, "url:", f"https://huggingface.co/datasets/{REPO}")
