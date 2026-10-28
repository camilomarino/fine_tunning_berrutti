# Datos, entrenamiento y resultados

Organización vigente desde el 28 de octubre de 2026. Las implementaciones se
separan por responsabilidad; los bundles ya enviados y sus hashes no cambian.

```text
data/                              catálogo, acceso e integridad
  berrutti_175/split-v1.json        partición original 125/25/25
  annotations.py                   cajas y orden de lectura auditado
  preparation/                     preparación CPU de corpus y sintéticas
archive_berrutti/                  descarga y biblioteca de rollos Archive
finetune/
  jobs.py                          preparación común de jobs Ruche/ClusterUY
  recipes/berrutti_qwen38/          receta Qwen, runtime y scripts Slurm
evaluation/scoring.py               política OCR, CER/WER y selección
reporting/berrutti/                 auditoría, estadísticas y publicación
  web/                             interfaz histórica de resultados
  ocr_diff_web/                    contexto del OCR Diff Viewer actual
reports/berrutti/                   informe final y figuras cerradas
tests/{data,training,evaluation,reporting}/
experiments/berrutti_qwen38/         compatibilidad con entradas anteriores
```

## Dependencias

- Datos y Archive pueden usarse sin importar entrenadores, Torch o viewers.
  El verificador del corpus, los hashes y las anotaciones pertenecen a datos.
- Entrenamiento consume datos sellados y las métricas de `evaluation/`.
  Cada receta concentra sus operaciones específicas; `finetune/jobs.py` conserva
  el wrapper común de Slurm y la integración genérica con TRL.
- Reportes lee evidencia cerrada. Puede consultar contratos de una receta, pero
  el entrenador no depende del código de publicación.
- `reports/` contiene informes y figuras revisables. Imágenes del corpus, OCR,
  checkpoints, logs, perfiles reales y bundles publicados siguen fuera de Git.

`tests/test_architecture.py` comprueba estas fronteras y la compatibilidad de
importaciones y comandos antiguos. Los alias apuntan al módulo real, de modo
que no duplican estado global. Los dos scripts Slurm vigentes y el split
conservan enlaces desde sus rutas antiguas. Los launchers retirados permanecen
retirados.

## Entradas actuales

```bash
python -m data agent --help
python -m archive_berrutti agent --help
python -m finetune --help
python -m finetune.recipes.berrutti_qwen38.three_arm_plan --help
python -m data.preparation.luisa --help
python -m reporting.berrutti.final_audit --help
python -m reporting.berrutti.benchmark_publication --help
make verify
```

La creación de un bundle con `python -m finetune` no envía trabajos. Los comandos
de preparación y auditoría deben recibir rutas privadas nuevas; sus `--help`
no leen datasets ni cargan un modelo.

## Próximo corpus y receta

El usuario confirmó el 28 de octubre de 2026 que el próximo corpus será de
**artículos de arXiv**. Tendrá preparación y receta propias, separadas de la
biblioteca Archive Berrutti existente. Todavía faltan definir la selección de
artículos, su representación (texto, PDF o imágenes de página), los targets y
la evaluación. No se registró ni descargó ese corpus ni se lanzó entrenamiento.

1. Registrar el corpus en `data/catalog.json` con procedencia, unidades y tipo
   de supervisión. Las ubicaciones reales van en `data/locations.local.json`,
   ignorado por Git.
2. Preparar un manifest nuevo, hashes y particiones por fuente/documento. Definir
   los targets y excluir la evaluación que deba quedar reservada.
3. Preparar una receta independiente en `finetune/recipes/` o una configuración
   privada de TRL. Reutilizar `finetune/jobs.py`; no copiar el publicador ni
   cambiar las fuentes congeladas de Berrutti.
4. Conservar configuración, semillas, identidad del modelo y criterios de
   evaluación. Antes de entrenar, aprobar los controles CPU y el ensayo GPU
   del destino conforme a `AGENTS.md`.
5. Añadir el análisis cerrado a `reporting/` y el informe final a `reports/`.

La preparación del corpus arXiv y su entrenamiento se harán en una operación
nueva, con entradas y resultados propios.
