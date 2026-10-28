# Auditoría, reportes y publicación

Este paquete lee resultados cerrados. No contiene el bucle de entrenamiento
ni la preparación de datasets.

En `berrutti/`:

- `final_audit.py` y `audit_optimizer_contract.py`: contratos de checkpoints,
  coberturas, intentos y métricas.
- `audit_statistics.py`: comparación pareada e incertidumbre descriptiva.
- `results_viewer.py` y `web/`: exportador e interfaz histórica.
- `benchmark_publication.py` y `ocr_diff_web/`: publicación del OCR Diff Viewer.
- `monitor_status.py` y `report_to_chat.py`: clientes históricos de consulta y
  entrega. Se conservan para compatibilidad; no crear ni reactivar programación.

```bash
python -m reporting.berrutti.final_audit --help
python -m reporting.berrutti.benchmark_publication --help
```

Los informes finales están en [reports/](../reports/README.md). Los bundles,
recibos detallados y transcripciones quedan en almacenamiento privado. La
publicación remota sigue el CLI transaccional del benchmark y su skill.
La reorganización no modifica la release que ya está publicada.
