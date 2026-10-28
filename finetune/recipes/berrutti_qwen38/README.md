# Tres experimentos Berrutti

La comparación de representaciones compara sólo páginas, sólo líneas y su combinación.
El [protocolo](../../../docs/THREE_ARMS_125_PROTOCOL.md) fija las 125 páginas train,
sus 5.212 líneas, tres épocas por modelo y validación/test de 25 páginas cada uno.
Los tres entrenamientos están cerrados: 48 updates y tres épocas por modelo.
La [auditoría final](../../../docs/FINETUNING_AUDIT_20261022.md) verifica los checkpoints,
la selección por validación y las métricas recalculadas. Combinado gana por el
criterio prefijado de validación; páginas ofrece menor coste observado. El test
de líneas conserva una página incompleta y su carácter provisional.

El seguimiento es sólo manual desde el 14 de octubre de 2026 por pedido del
usuario. Los timers de consulta, informes al chat y vigilancia fueron retirados;
no programar llamadas al modelo ni reactivarlos sin una nueva petición explícita.

## Sellar las entradas

```bash
python -m finetune.recipes.berrutti_qwen38.three_arm_plan \
  --dataset /ruta/privada/berrutti/raw \
  --split data/berrutti_175/split-v1.json \
  --annotations /ruta/privada/bounding_boxes_lines_via2.json \
  --output /ruta/privada/nueva-operacion/inputs
```

El comando comprueba hashes, split y cajas de líneas, conserva todas las páginas
y etiquetas humanas, registra las discrepancias y crea `cohort.json` y `plan.json`
en una carpeta privada nueva. No lanza entrenamiento ni reserva GPU.

## Código de la receta

- `runner.py`, `luisa_mix.py` y `run_*.sbatch`: entrenamiento, evaluación por
  época y continuaciones con checkpoints. Las operaciones GPU se conservan.
- `three_arm_plan.py`: diseño de los tres brazos. El split está en
  `data/berrutti_175/split-v1.json`.
- `generation_guard.py` y `qwen_sampling.py`: generación acotada y política
  de muestreo de la extensión.
- `comparison.py`, `projection_runtime.py`, `eager_hooks.py`, `linear_decode.py`
  y `cuda_transport.py`: comprobaciones e integración del runtime Qwen.
- `performance.py` y `performance_assay.py`: caché de crops y ensayos técnicos.

La preparación CPU de LUISA y sintéticas está en `data/preparation/`.
La carga y las anotaciones Berrutti están en `data/berrutti.py` y
`data/annotations.py`. CER/WER y selección por validación están en
`evaluation/scoring.py`. Auditoría, publicación y viewers están en
`reporting/berrutti/`; los informes cerrados en `reports/berrutti/`.
Estos módulos no forman parte del entrenador.

El runner recorre todos los ejemplos durante tres épocas. Guarda un checkpoint
por actualización y restaura adaptador, optimizador, calendario y RNG cuando
continúa en otro job. Selecciona entre las tres validaciones completas y evalúa
test sólo para el checkpoint elegido. El calendario prevé 350 evaluaciones de
página sin reintentos, frente a 500 con test después de cada época.

La caché de 5.212 crops ya está preparada y verificada. La cuenta permite cuatro
A100 simultáneas: dos trabajadores de dos GPU, con disponibilidad sujeta a cola.
El perfil común usa BF16, SDPA y proyección visual FP32, sin caché de embeddings
visuales. Copia las mismas librerías a SSD local: el diagnóstico CPU midió
51 s de importación frente a 505 s desde el entorno compartido; incluyendo
la copia, el job duró 2 min 18 s. El ensayo GPU comprueba este perfil completo
antes de habilitar entrenamientos. Esas medidas no estiman el tiempo total.

Para cada operación se congelan fuentes, configuración e inputs privados nuevos.
`python -m finetune.recipes.berrutti_qwen38.runner --config CONFIG --root DESTINO --mode preflight`
comprueba perfil de runtime, hashes de fuentes, cohorte, splits y recibo de
crops sin cargar Torch ni reservar GPU. El perfil válido se declara en
`runner.RUNTIME_PROFILE`; se rechazan recetas con un perfil diferente antes
de cargar pesos. Esa comprobación usa la misma validación de entradas que el
entrenamiento y debe preceder al ensayo GPU en la cadena Slurm.
El ensayo prueba las tres estrategias en dos páginas de origen, restaura un
checkpoint tras cada actualización y genera una página de validación y una de
test. Requiere generación completa, gradientes finitos y restauración exacta.
Baseline depende de su éxito; los tres modelos dependen del baseline. Cada
trabajador solicita dos A100, ocho CPU y 96 GB; las continuaciones se reservan
al alcanzar el presupuesto de 21 horas, con un máximo de ocho segmentos.

Las comprobaciones de rendimiento se siguen aparte, sin ser experimentos
adicionales. LUISA queda fuera de estos tres brazos.

## Experimento cuatro autorizado

La extensión [Berrutti–LUISA con páginas sintéticas](../../../docs/EXPERIMENT_4_LUISA_SYNTHETIC.md)
incorpora todos los datos solicitados en tres épocas mixtas y una cuarta de
consolidación Berrutti. Completó 64 updates y seleccionó la época 4 por validación
completa: CER 2,048 %, con test descriptivo completo de 3,220 %. Conserva
inicialización nueva y validación de 25 páginas; tiene ensayo GPU, baseline y
checkpoints propios. Su generación usa muestreo instruct, diferente de los tres
brazos originales; la comparación directa no aísla el efecto de LUISA.
`luisa_prepare.py`, `synthetic_pages.py`, `luisa_mix.py` y `run_luisa_mix.sbatch`
implementan auditoría, síntesis, ejecución y continuaciones. El modo literal
con todo LUISA incluye solapamientos conocidos: sus CER son exploratorios y
no acreditan independencia respecto al test. Se sigue como extensión separada.
Los siguientes párrafos describen antecedentes del runtime; el entrenamiento
final y sus resultados están cerrados en el informe de auditoría.
Su candidata de memoria mantiene los momentos AdamW en RAM entre actualizaciones
y los devuelve al dispositivo del parámetro antes de cada paso, sin cambiar
dtype ni valores. El modelo y backward permanecen en dos A100. El ensayo exige
identidad exacta frente a AdamW residente, las tres actualizaciones Qwen
diagnósticas con restauración y los mismos cinco probes completos; ese control
de memoria no sustituye generación ni habilita por sí solo el entrenamiento.
La candidata siguiente mantiene los pesos almacenados y el entrenamiento;
prueba la proyección de salida FP32 durante generación para investigar la
repetición persistente. Registra controles de precisión sin cambiar política,
probes ni intentos, y sólo el ensayo completo puede habilitar su propio baseline.
El ensayo de proyección FP32 no resolvió la repetición de la página de dos
columnas. La candidata posterior prueba atención SDPA matemática FP32 del
decodificador durante inferencia, con controles FP64 acotados y restauración
de la interfaz. Mantiene pesos, cachés, entrenamiento y generación; sigue
necesitando el gate GPU completo antes de lanzar baseline o entrenamiento.


El primer ensayo de atención falló antes de generar por el nombre del argumento
de máscara de la interfaz Qwen. La corrección usa `attention_mask`, verifica
las llamadas por nombre y conserva las mismas operaciones, probes y controles.
No acredita todavía la precisión ni habilita baseline o entrenamiento.

La interfaz corregida completó los tres updates y controles de atención/proyección,
pero la misma composición de dos columnas repitió ambos intentos, sin OOM.
La candidata siguiente prueba acumulación FP32 sólo en la convolución causal
durante inferencia, conservando pesos y estado BF16 y todas las condiciones de
generación. Añade controles FP64 y de actualización exacta de caché, restaura
las interfaces al salir y exige un nuevo ensayo completo antes de habilitar la
extensión. Los resultados y fuentes anteriores permanecen intactos.

El control posterior sobre tres heads capturados confirmó un desacuerdo del
chunk FLA frente a la referencia FP64 y una coincidencia mucho mayor del cuerpo
PyTorch nativo. `linear_decode.py` prueba ese cuerpo en FP32 sólo para prefill de
inferencia, manteniendo salida BF16, estado FP32 y FLA recurrente con caché.
Conserva todas las operaciones de entrenamiento y condiciones del ensayo.
Los controles numéricos y de interfaces se añaden al gate previo, sin sustituir
las tres actualizaciones diagnósticas, restauraciones ni cinco generaciones.
La corrección sigue siendo candidata hasta completar el ensayo GPU íntegro.

## Destino ClusterUY

La comparación de Ruche conserva su copia congelada. Una operación nueva puede
seleccionar `gpu_type: "A40"` en el config del runner y solicitar dos A40 en un
nodo ClusterUY. Los datos, pesos, semillas, píxeles, pérdidas y evaluación se
mantienen; la identidad del runtime incluye los nombres de GPU y requiere
baseline y ensayo propios. No reutilizar el baseline de otro hardware.

`run_three_arms.sbatch` admite como sexto argumento la imagen Singularity de
ClusterUY. Usa scratch del job, transmite la máscara GPU asignada por Slurm y
mantiene un proceso host de inicialización CUDA; lo termina al salir. Las
continuaciones mantienen partición normal, QoS GPU y dos A40. Los recursos
ClusterUY se indican a `sbatch`, sobreescribiendo los defaults Ruche del archivo.
El sexto argumento vacío conserva la ejecución nativa de Ruche.

El ensayo representativo de Qwen completo en dos A40 cerró con Slurm COMPLETED
y recibo exitoso, incluidas las tres actualizaciones diagnósticas, restauración
y seis generaciones completas. Acredita compatibilidad de ese ensayo, sin
extrapolar calidad ni velocidad. Los tres modelos originales finalizaron en
Ruche: no hay modelos pendientes de migración ni se envían duplicados.
