# Recetas de entrenamiento

Cada receta contiene su entrenador específico y sus ensayos, separado de datos
y publicación. La infraestructura común de Slurm y TRL está en `finetune/jobs.py`.

- [Qwen–Berrutti](berrutti_qwen38/README.md): tres estrategias Berrutti y la
  extensión LUISA separada, con sus políticas cerradas.

Para otro corpus, consumir sus manifests desde `data/` y añadir una receta nueva
o una configuración de TRL. Las ubicaciones reales y las configuraciones de
cómputo se guardan fuera de Git. No reutilizar automáticamente las particiones,
targets ni operaciones específicas Qwen del experimento anterior.
