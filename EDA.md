# EDA — UNSW-NB15

Reproducible con `python src/eda.py` (pandas 3.0.6, Python 3.14). Datos: `UNSW_NB15_training-set.csv` y `UNSW_NB15_testing-set.csv` (partición oficial de UNSW; versión exacta por documentar con DVC).

## Estructura
- Train: 175.341 filas · Test: 82.332 filas · 45 columnas idénticas (30 int, 11 float, 4 texto).
- **Cero valores nulos** en ambos archivos.
- Objetivo: `label` (0 normal / 1 ataque). `attack_cat` (10 clases) es la ampliación y **no puede usarse como feature** (revela la respuesta). `id` es solo un índice 1..N.

## Balance de clases
| | Normal | Ataque |
|---|---|---|
| Train | 31,9 % | 68,1 % |
| Test | 44,9 % | 55,1 % |

Familias muy desiguales: Generic 22,8 %, Exploits 19,0 %, Fuzzers 10,4 %, DoS 7,0 %, Reconnaissance 6,0 %, Analysis 1,1 %, Backdoor 1,0 %, Shellcode 0,65 %, **Worms 0,07 % (130 filas)**.
La proporción cambia entre train y test: el umbral elegido en validación no tiene por qué ser óptimo en test.

## Hallazgos que condicionan el diseño
1. **`id` filtra la respuesta.** Las filas están ordenadas por tipo: en train, el 0–35 % de los id es 100 % normal y el 78–100 % es casi todo ataque; en test el patrón es distinto (bloques de 0 y 1). Usarlo como feature, o partir train/val por rangos de id, daría resultados engañosos. **Excluir `id`** y partir con mezcla aleatoria estratificada.
2. **Duplicados masivos.** Filas con features idénticas a otra fila (sin `id`, `label` ni `attack_cat`): 74.301 en train (42,4 %) y 28.386 en test (34,5 %). Se concentran en ataques: 58,8 % de los ataques de train están repetidos frente a 7,5 % de las normales (en test, 56,4 % y 7,6 %). Si no se deduplican antes de partir, el mismo ejemplo cae en train y validación y se infla el rendimiento.
3. **Etiquetas contradictorias.** 229 grupos de features idénticas aparecen con `label` distinto en train (1.772 con `attack_cat` distinta): hay un techo de rendimiento por ruido de etiquetas.
4. **Solapamiento train/test.** 8.541 filas de test (10,4 %) tienen features exactamente iguales a alguna de train. Es una fuga a documentar al evaluar.
5. **Variables casi deterministas.** `sttl` (|r|=0,69 con label) y `ct_state_ttl` (0,58): valores de TTL concretos separan casi perfectamente (sttl=31 → 0 % ataque en 39.455 filas; sttl=254 → 90 %). Reflejan cómo se generó el laboratorio (hosts/IPs fijos), no necesariamente rasgos de ataque en una red real. Mencionar en limitaciones.
6. **Categorías desalineadas.** `proto`: 133 valores en train, 131 en test (`icmp`, `rtp` solo en train). `state`: `ACC` y `CLO` solo en test; `ECO`, `PAR`, `URN`, `no` solo en train. Se necesita codificación que tolere categorías no vistas. `proto='unas'` (12.084 filas) y `sctp` son 100 % ataque; `arp` 0 %.
7. **Colas muy pesadas.** Asimetría extrema en `trans_depth` (167), `response_body_len` (76), `sbytes` (45), `sloss`, `dloss`, `spkts`, `dbytes`… y 40–99 % de ceros en muchas. Los modelos lineales necesitarán `log1p` y escalado; los árboles no.
8. **Colinealidad.** 12 pares con |r|>0,95: `is_ftp_login`≈`ct_ftp_cmd` (1,0; son idénticas), `sbytes`~`sloss`, `dbytes`~`dloss`, `swin`~`dwin`, `ct_srv_src`~`ct_srv_dst`. Redundancia relevante solo para modelos lineales/interpretación.
9. **Variables constantes o casi:** ninguna constante; `is_sm_ips_ports` binaria con 98 % de ceros.
10. **Desplazamiento train→test moderado:** máx. 0,19 desviaciones estándar (`synack`, `tcprtt`, `ackdat`), sin cambios drásticos en las medias.

## Decisiones derivadas (por validar en los experimentos)
- Features: las 42 columnas restantes (sin `id`, `label`, `attack_cat`).
- Deduplicar por features **antes** de particionar y registrar cuántas filas se pierden (cambia la población estudiada).
- Partición: validación estratificada a partir de train; test oficial reservado para una sola evaluación final.
- Métricas: recall, precision, FPR y macro-F1; no accuracy sola.
- Preprocesamiento dentro del pipeline guardado: one-hot con `handle_unknown='ignore'` para `proto/service/state`, `log1p` para los modelos lineales.
