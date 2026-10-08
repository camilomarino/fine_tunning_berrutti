# Guía del proyecto de fine-tuning Berrutti

- Este directorio es el hogar del entrenamiento Qwen–Berrutti. El repositorio
  `berrutti_benchmark` conserva el benchmark, sus registros y su publicación.
- Usar la skill `ruche` para Slurm. Entrenar solo en Ruche, nunca en ClusterUY
  ni en los nodos de login. No detener trabajos ajenos ni las corridas previas.
- Dataset, crops, OCR, checkpoints, entornos, tokens, logs y evidencia privada
  quedan fuera de Git. `.private/` está ignorado; no copiar datos licenciados a
  archivos versionados.
- Cada nueva corrida, prueba y checkpoint utiliza un identificador y ruta
  nuevos. Los resultados heredados son evidencia inmutable.
- Antes de una corrida grande: prueba de asignación GPU, smoke de runtime,
  ensayo con pocas páginas, luego entrenamiento completo. Verificar el par de
  A100 **dentro** del job. Guardar configuración, semillas, hashes y eventos.
- Usar la misma política de generación y puntuación para comparar baseline y
  checkpoints. Seleccionar por CER de validación completa; el test es
  descriptivo y nunca decide un checkpoint. El test de la época 2 de la
  corrida heredada está incompleto y su CER es provisional.
- Mantener detección acotada de repetición, como máximo dos intentos por
  página, y conservar el estado de páginas incompletas. No ocultar fallos de
  CUDA, pérdida no finita ni gradientes no finitos.
- Hacer `make verify` antes de entregar cambios. Para cambios de runtime GPU,
  agregar además un smoke bajo Slurm, sin alterar una corrida activa.
- Leer `docs/HANDOFF.md` y el README de la receta antes de relanzar el
  currículo. El chat original sigue vinculado al worktree del benchmark; este
  archivo preserva el contexto para chats abiertos desde este directorio.

