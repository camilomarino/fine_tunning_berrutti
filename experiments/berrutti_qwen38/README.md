# Entradas históricas Qwen–Berrutti

Las implementaciones se separaron por responsabilidad. Este directorio conserva
únicamente alias de importación y comandos antiguos; no añadir código nuevo aquí.

- [Entrenamiento](../../finetune/recipes/berrutti_qwen38/README.md):
  `finetune.recipes.berrutti_qwen38`.
- [Preparación de datos](../../data/preparation/README.md): `data.preparation`.
- [Evaluación compartida](../../evaluation/README.md): `evaluation.scoring`.
- [Auditoría y publicación](../../reporting/README.md): `reporting.berrutti`.
- [Informes cerrados](../../reports/README.md): `reports/berrutti/`.

Los módulos antiguos apuntan al mismo objeto Python que la implementación nueva,
para mantener importaciones, comandos y parches existentes. Las rutas antiguas de
los dos scripts vigentes y del split son enlaces a sus ubicaciones actuales.
No se restauraron launchers retirados ni se modificaron bundles congelados.
