# Guía del proyecto de fine-tuning Berrutti

- Desde el 14 de octubre de 2026, el seguimiento del entrenamiento es sólo
  manual por pedido explícito del usuario, en todos los chats de este repositorio.
  Se retiraron los informes periódicos y la vigilancia que llamaban al modelo.
  No crear ni reactivar cron, timers, automatizaciones, heartbeats o avisos al
  modelo sin una nueva petición explícita. Consultar el estado sólo cuando lo
  pida el usuario. Los entrenamientos Slurm y sus continuaciones nativas siguen
  ejecutándose; esta desactivación no los cancela.
- Este directorio es el hogar del entrenamiento Qwen–Berrutti. El repositorio
  `berrutti_benchmark` conserva el benchmark, sus registros y su publicación.
- Separar responsabilidades según `docs/REPOSITORY_STRUCTURE.md`: datos en
  `data/` y `archive_berrutti/`; entrenamientos en `finetune/recipes/` sobre
  el launcher común `finetune/`; métricas en `evaluation/`; auditoría,
  publicación y viewers en `reporting/`; informes finales en `reports/`.
  Datos no debe importar entrenamiento ni reportes; entrenamiento no debe
  importar publicación. `experiments/berrutti_qwen38/` conserva sólo entradas
  de compatibilidad: no añadir implementaciones nuevas ahí. Las copias
  congeladas y la publicación vigente permanecen intactas.
- Usar las skills `ruche` o `cluster-uy` según el destino Slurm. El usuario
  autorizó preparar interoperabilidad entre ambos clusters. Entrenar sólo en
  nodos de cómputo, nunca en login. No detener trabajos ajenos. Cancelar corridas propias
  y purgar sus resultados sólo cuando el usuario lo solicite explícitamente.
- Dataset, crops, OCR, checkpoints, entornos, tokens, logs y evidencia privada
  quedan fuera de Git. `.private/` está ignorado; no copiar datos licenciados a
  archivos versionados.
- La entrada común a los datasets es `data/README.md` y `data/catalog.json`.
  `python -m data list/show/path` consulta las ubicaciones privadas sin reservar
  CPU. Los payloads bajo `data/*/releases/` y `data/locations.local.json` están
  ignorados. `.private/dataset` mantiene compatibilidad con Berrutti 175.
  LUISA conserva sus particiones de origen y no se mezcla automáticamente con
  el entrenamiento: hay solapamiento detectado con páginas de test Berrutti.
- La entrada unificada es `berrutti-data` / `python -m data`: menú con flechas
  para humanos y `agent list/info/files/text/annotations` para agentes. Ver
  `data/CLI.md` para filtros, paginación, rutas y contrato JSON. Los índices
  supervisados se cachean en `.private/data-browser/`; `agent refresh CONJUNTO`
  relee metadatos sin modificar originales. En Archive, la misma entrada delega
  descarga, procesamiento y trabajos al motor existente. Berrutti 175 conserva
  8.975 cajas con texto; su orden VIA2 no garantiza el orden de lectura del TXT.
- Para consultar o preparar la biblioteca Archive, usar `archive-berrutti agent`
  (o `python -m archive_berrutti agent`): comandos sin TTY, JSON versionado,
  rutas absolutas y trabajos consultables por ID. Ver el contrato y los ejemplos
  en `archive_berrutti/README.md`, sección «Uso por agentes».
- La biblioteca predeterminada de esta máquina es el perfil SSH/Slurm `ruche`,
  en `$WORKDIR/berrutti-data/archive/library-v1`. Leer
  `archive_berrutti/REMOTE_RUCHE.md` y consultar `archive-berrutti profile show`.
  Usar `--local` para el disco local; `--root` sin `--profile` conserva su
  significado local. Para tareas remotas largas usar `agent ... --background`;
  consultar su ID con `agent job`. No reenviar trabajos que estén en cola.
  Los perfiles y las rutas privadas no se versionan. Actualizar código con
  `profile deploy ruche`, sin alterar versiones ni entornos de jobs existentes.
  El explorador y `agent list/info/files/plan/job` leen el índice por SSH sin
  reservar CPU. `agent index` lo reconstruye bajo Slurm si falta o hubo cambios
  externos; los trabajadores nuevos lo actualizan durante el procesamiento.
- Cada nueva corrida, prueba y checkpoint utiliza un identificador y ruta
  nuevos. No sobrescribir resultados; una limpieza explícitamente solicitada
  puede retirar recetas obsoletas y borrar sus resultados.
- Antes de una corrida grande: prueba de asignación GPU, smoke de runtime,
  ensayo representativo, luego entrenamiento completo. Verificar las GPU del
  perfil **dentro** del job; Ruche usa dos A100 y el destino ClusterUY exige
  aprobar un ensayo nuevo con dos A40 antes de migrar el entrenamiento.
  Guardar configuración, semillas, hashes y eventos.
- `python -m finetune` prepara bundles Slurm para Ruche/ClusterUY sin enviar
  trabajos. Ver `finetune/README.md`: TRL es el motor genérico; el runner Qwen
  conserva sus operaciones específicas. Un bundle preparado no acredita
  compatibilidad GPU. No cambiar fuentes/entornos congelados de jobs enviados.
- Usar la misma política de generación y puntuación para comparar baseline y
  checkpoints. Seleccionar por CER de validación completa; el test es
  descriptivo y nunca decide un checkpoint. Cualquier CER con páginas
  incompletas es provisional y no acredita una mejora válida.
- Mantener detección acotada de repetición, como máximo dos intentos por
  página, y conservar el estado de páginas incompletas. No ocultar fallos de
  CUDA, pérdida no finita ni gradientes no finitos.
- Hacer `make verify` antes de entregar cambios. Para cambios de runtime GPU,
  agregar además un smoke bajo Slurm, sin alterar una corrida activa.
- La comparación de representaciones está en `docs/THREE_ARMS_125_PROTOCOL.md`:
  tres modelos Berrutti, las 125 páginas train y tres épocas completas.
  LUISA queda fuera de esos tres brazos. El usuario autorizó por separado el
  experimento cuatro: todas las líneas LUISA, Berrutti train completo y páginas
  sintéticas, con una etapa adicional de consolidación. Ver
  `docs/EXPERIMENT_4_LUISA_SYNTHETIC.md`. No mezclar sus resultados con la
  comparación controlada ni ocultar la contaminación conocida del test.
  No restaurar launchers retirados.
- Leer `docs/HANDOFF.md` y el README de la receta antes de preparar una corrida.
