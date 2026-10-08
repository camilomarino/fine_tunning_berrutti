# Traspaso del experimento Qwen–Berrutti

Este proyecto procede del chat y worktree de `berrutti_benchmark` del 7–8 de
octubre de 2026. El propósito del nuevo directorio es desarrollar el
fine-tuning por separado; el benchmark y su worktree siguen intactos.

## Decisiones vigentes

- Modelo: revisión fija de Qwen3.8-27B; BF16 en dos A100 de 40 GB, LoRA r16,
  alpha32 y dropout 0,1 sobre el decodificador; encoder visual congelado.
- Dataset: Berrutti v3, 175 páginas, split provisional agrupado por rollo
  125/25/25. Falta auditar identidad documental entre rollos. Ningún rollo
  cruza conjuntos y no se detectaron duplicados exactos ni candidatos de
  5-gramas con similitud ≥0,60 entre conjuntos.
- Prompt: transcripción fiel en orden de lectura, conservando saltos de línea,
  sin explicaciones ni correcciones. La política `ascii-text-v1` ignora las
  diferencias de saltos de línea al calcular CER/WER.
- Evaluar validación y test antes de entrenar y después de cada época. Elegir
  checkpoint solo por validación; el test se usa como tendencia descriptiva.
- Monitorizar e informar en el chat cada dos horas. Los temporizadores systemd
  del usuario cumplen esa función; la vista Scheduled nativa de Codex no quedó
  habilitada en esta sesión.
- No interrumpir otros trabajos. Usar solo Ruche para entrenamiento.

## Evidencia heredada

La corrida principal completó tres épocas. CER de validación: línea base
2,746 %, épocas 1/2/3: 2,380 %, 2,330 %, 2,184 %. CER de test: línea base
5,027 %, época 1: 3,791 %, época 3: 3,437 %. El 10,716 % de test de época 2
incluye una página incompleta y no acredita empeoramiento. Las guardas
detectaron bucles repetitivos y reintentaron una vez por página. El ensayo
de dos imágenes completó cuatro épocas; su mejora es solo un control técnico.

La prueba pequeña del currículo falló antes de cargar el modelo. El log muestra
que CUDA/NVML no pudo inicializarse y PyTorch vio cero GPUs, aunque Slurm
había reservado dos. La corrida grande del currículo sigue pendiente por una
dependencia que no se cumplió. Próximo paso: inspeccionar la asignación
fallida, ejecutar un smoke de visibilidad de las dos GPUs en una asignación
nueva y reintentar el ensayo en un directorio nuevo. No modificar la evidencia
fallida ni detener los trabajos pendientes sin una decisión explícita.

## Continuidad técnica

Los scripts y el split están en `experiments/berrutti_qwen38/`. La copia del
manifiesto de crops, probes, recibos de staging y un snapshot de la conversación
original (`.private/chat/conversation.md` y su JSONL) están en `.private/`;
este directorio se ignora en Git. Los artefactos
completos y checkpoints permanecen en Ruche, identificados por la copia
privada de `coordination.json`. El monitor activo conserva el destino del chat
original hasta que exista una nueva conversación en este proyecto. Los dos
servicios systemd del usuario ya apuntan a los scripts de este directorio;
los temporizadores continúan activos. Las unidades anteriores y el último
snapshot están respaldados en `.private/monitor/`. Al abrir un nuevo chat,
actualizar sólo el `--thread-id` del servicio de informes, recargar systemd y
comprobar ambos temporizadores. No modificar jobs ni evidencia remota.

Este archivo es un resumen operativo, no un traslado nativo del chat en la
aplicación. La conversación original conserva su historial y su cwd antiguo;
los chats nuevos pueden usar este documento y la copia privada del transcript
para continuar el trabajo desde este directorio.
