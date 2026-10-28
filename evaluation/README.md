# Evaluación compartida

`scoring.py` conserva exactamente `ascii-text-v1`: AnyASCII seguido de
normalización de espacios, distancia Levenshtein y CER/WER microagregados.
Incluye la selección por validación completa, con desempate por época.

Lo consumen entrenamiento y auditoría. No importa datasets, modelos, GPU ni
herramientas de publicación. Las métricas del experimento Berrutti no cambian
por esta separación.
