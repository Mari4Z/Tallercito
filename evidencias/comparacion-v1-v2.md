# Comparación v1 / v2 (solo validación)

v2 cambia **dos cosas a la vez**: el modelo (regresión logística → HistGradientBoosting) y el umbral (0.5 → 0.65). Se separan así, con la validación:
- **Solo cambiar el modelo** (boosting con umbral 0.5): recall 0.9754 (igual que v1: 0.9753), FPR 0.0832 (v1: 0.1326), macro-F1 0.9498 (v1: 0.9304). Mejora las falsas alarmas sin perder ataques.
- **Además subir el umbral a 0.65:** FPR baja a 0.0487 a cambio de recall 0.9532.

| Métrica (validación) | v1 · logística · umbral 0.5 | v2 · boosting · umbral 0.65 | Cambio |
|---|---|---|---|
| Recall (ataques detectados) | 0.9753 | 0.9532 | -0.0221 |
| Precision (alertas correctas) | 0.9400 | 0.9766 | +0.0366 |
| FPR (tráfico normal alarmado) | 0.1326 | 0.0487 | -0.0839 |
| Macro-F1 | 0.9304 | 0.9462 | +0.0158 |
| Ataques perdidos (FN) | 590 | 1117 | +527 (+89 %) |
| Falsas alarmas (FP) | 1485 | 545 | -940 (-63 %) |

**Mejora:** menos falsas alarmas (FPR 0.133 → 0.049), más precision y mayor macro-F1, con un modelo de 2 MB.
**Empeora:** el recall baja 2.2 puntos, es decir, se pierden 527 ataques más en validación.
**Error que se acepta:** dejar pasar algunos ataques más (FN 590 → 1117) para recibir menos falsas alarmas, con un máximo de 1 por cada 20 conexiones normales. Se asume que un equipo de seguridad satura con demasiadas alertas falsas.

