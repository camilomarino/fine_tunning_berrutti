# Fine-tuning en Ruche y ClusterUY

`python -m finetune` prepara un bundle privado con un único wrapper Slurm para
los dos clusters. La receta elige el entrenador; el perfil elige los recursos
y el entorno. Preparar un bundle no instala librerías, mueve datos, consulta
la cola ni envía trabajos. Su estado es `PREPARED_NOT_SUBMITTED`.

## Qué se comparte y qué depende de la tarea

| Capa | Implementación | Qué cambia al elegir otro modelo o cluster |
| --- | --- | --- |
| Trabajo Slurm | `finetune/jobs.py` | Perfil privado: cuenta, partición, GPU, memoria, tiempo, entorno y rutas |
| Fine-tuning general SFT | CLI oficial de TRL, sobre Transformers/PEFT/Accelerate | Configuración YAML: modelo fijado, datos, LoRA, precisión y paralelismo |
| Receta propia | Motor `command`, argumentos estructurados | Entry point del proyecto, sin copiar un launcher por cluster |
| Comparación Berrutti vigente | `finetune/recipes/berrutti_qwen38/runner.py` | Conserva agrupación por página, pérdida 50/50, CER, checkpoints y sus ajustes Qwen |

La [CLI de TRL](https://huggingface.co/docs/trl/clis) evita implementar otro bucle
de entrenamiento. Su [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
admite texto conversacional y modelos compatibles de imagen más texto. Así se
pueden preparar recetas para Gemma, otras versiones Qwen u otras arquitecturas
soportadas por las versiones instaladas, sin clases vacías ni un registro por
marca. Un LLM de texto recibe texto; para recibir páginas o recortes hace falta
una variante visual y su processor. El tamaño depende de la VRAM y del reparto
de pesos, gradientes y estado del optimizador: LoRA no elimina los pesos base.

Cambiar sólo el nombre del modelo en el runner Berrutti actual **no funciona**:
fija una clase Qwen, targets LoRA, proyección visual FP32, routing, presupuesto
de píxeles y `logits_to_keep`. Esas operaciones no se aplican automáticamente a
Gemma u otra familia. Tampoco se reemplaza la pérdida por página por la pérdida
por ejemplo de TRL: produciría otro experimento. Las fuentes y el entorno de los
tres trabajos ya enviados permanecen congelados.

## Preparar una receta

Guardar perfiles y recetas reales bajo `.private/finetune-config/`. Este ejemplo
de perfil debe completarse con una cuenta y rutas autorizadas, verificadas en
Slurm; no acredita disponibilidad, cuota ni prioridad:

```json
{
  "cluster": "ruche",
  "partition": "gpua100",
  "account": "CUENTA_AUTORIZADA",
  "gpu_type": "a100",
  "gpus": 1,
  "cpus": 8,
  "memory_gb": 96,
  "scratch_gb": 30,
  "time": "02:00:00",
  "workdir": "/ruta/privada/codigo-fijado"
}
```

Para ClusterUY, usar `cluster: "clusteruy"`, la partición/QoS autorizadas y
`gpu_type: "a40"` o el tipo que se haya comprobado. El wrapper solicita GRES
sin tipo en Ruche y con tipo en ClusterUY; usa el `$TMPDIR` de Slurm en Ruche y
`/scratch/$USER/$SLURM_JOB_ID` en ClusterUY. Todos los trabajos son de un nodo y
una tarea Slurm. El entrenador puede crear procesos GPU dentro de esa tarea.

Los [recursos publicados de ClusterUY](https://www.cluster.uy/ayuda/recursos_disponibles/)
incluyen dos A40 por máquina y una A100 por máquina. No se puede trasladar sin
cambios un programa que requiere dos A100 locales a sus dos A100 separadas.
Los límites efectivos y la ocupación requieren una consulta nueva por SSH.

El perfil puede añadir `qos`, un mapa `environment` de variables explícitas o:

```json
"container": {
  "image": "/ruta/privada/runtime-fijado.sif",
  "binds": ["/ruta/privada/modelos", "/ruta/privada/datos"]
}
```

Se usa Apptainer en Ruche y Singularity en ClusterUY, con `--nv --cleanenv`, binds
explícitos y caché temporal local. Los ejecutables de la receta son los del
interior del contenedor cuando se usa uno. La imagen, driver, CUDA y librerías
deben probarse en el destino; copiar un virtualenv entre sistemas no basta.
La máscara `CUDA_VISIBLE_DEVICES` asignada por Slurm se transmite explícitamente
al contenedor; el perfil no puede reemplazarla. Se admiten cuentas con puntos,
como las asociaciones institucionales de ClusterUY.

Una receta TRL sólo añade este JSON; `config` se resuelve respecto a él:

```json
{
  "engine": "trl-sft",
  "executable": "/ruta/privada/entorno/bin/trl",
  "config": "sft.yaml"
}
```

En `sft.yaml` se usan directamente los argumentos oficiales de la versión TRL
fijada: `model_name_or_path`, `model_revision`, datos, semillas, LoRA, etc. No
se replica su parser aquí. Fijar también versiones de dependencias y revisiones
de datos. Para una receta genérica, indicar `dataset_test_split: validation` si
se evalúa durante el entrenamiento: ese nombre de argumento TRL no convierte
el conjunto de test en validación. La evaluación por loss de TRL no es CER.
El CER y la selección por validación completa siguen en la receta OCR propia.
No subir modelos/datos privados al Hub desde la configuración de entrenamiento.

Para datos visuales, verificar plantilla de chat, máscara de supervisión,
tokens de imagen y `max_length`; la documentación TRL recomienda `None` para
evitar truncar tokens visuales. Para varias GPU, configurar Accelerate/FSDP o
ZeRO según memoria y modelo. DDP replica el modelo; reservar dos GPU no implica
que sus memorias se sumen. El wrapper no decide precisión, cuantización,
resolución, packing o kernels. Esas decisiones requieren medir calidad y tiempo.

```bash
python -m finetune \
  --profile .private/finetune-config/ruche.json \
  --recipe .private/finetune-config/recipe.json \
  --run-dir /ruta/privada/operaciones/NUEVA_OPERACION
```

Devuelve un directorio local privado con `job.sh`, la configuración TRL copiada
sin modificar, los JSON de origen, hashes y manifiesto. Rechaza sobrescrituras,
valores Slurm ambiguos y argumentos shell implícitos. El `output_dir` TRL se
fija dentro de la operación nueva. `--output` permite elegir otra carpeta local
privada nueva. El bundle se copia íntegro al `--run-dir` del destino; modelo,
datasets, entorno y código ya fijado se preparan aparte. Transferencias grandes
a ClusterUY deben usar su puerto de transferencia documentado, 10022.

Para una receta propia se sustituye el JSON anterior por:

```json
{
  "engine": "command",
  "argv": ["/ruta/privada/python", "-m", "paquete.receta", "--output", "{run_dir}/resultados"]
}
```

Los argumentos se citan individualmente; `{run_dir}` es la única sustitución.
Las continuaciones, checksums de código/datos/modelo, checkpoints y el grafo
de dependencias pertenecen a la receta. No se crean continuaciones automáticas
para comandos arbitrarios. El runner Berrutti ya implementa estas garantías;
su operación enviada sigue usando su launcher congelado, sin migración automática.
El monitor de Berrutti usa el campo `cluster` de su coordinación privada
(`Ruche` o `ClusterUY`); nunca consulta identificadores Slurm de un destino en
el otro. Cambiar esa coordinación requiere antes verificar el ensayo del destino.

## Estado y pruebas antes de ejecutar

El script exige Slurm, comprueba los hashes del bundle dentro del job y registra
inicio y exit code sin sobrescribir una ejecución anterior. `process.json` sólo
acredita el exit code del proceso: no sustituye Slurm `COMPLETED`, el recibo del
entrenador ni sus métricas. Un fallo no se convierte en entrenamiento exitoso.

Antes de lanzar una combinación nueva de modelo/cluster: comprobar cuenta,
QoS, recursos y `sbatch --test-only`; luego prueba CUDA host/contenedor bajo
Slurm, forward/backward, checkpoint con optimizador/RNG y generación en una
muestra representativa. Finalmente medir duración y memoria. No reutilizar un
baseline o checkpoint entre perfiles sin verificar identidad y compatibilidad.

**Implementado y probado en CPU:** preparación, sintaxis y ejecución del wrapper,
argumentos literales, hashes, propagación de errores y prevención de reejecución.
En ClusterUY se verificaron por Slurm la construcción del contenedor Python
3.12 y CUDA host/contenedor en una A100, ambos con recibo exitoso. Eso comprueba
acceso y exposición CUDA, sin acreditar todavía entrenamiento de un modelo.
También terminó bajo Slurm la importación CPU de Torch, Transformers, PEFT y
Accelerate, con hash del runtime y datos verificados. El ensayo sintético
Gemma/Qwen está en cola y el representativo Qwen de dos páginas espera dos A40.
Los recibos y el candidato vigente están en la operación privada indicada por
`.private/clusteruy-current-setup.json`; su `submission.json` conserva el envío
inicial y `memory-candidate.json` identifica el ensayo vigente. Las recetas TRL nuevas
siguen requiriendo su propio ensayo; no se afirma compatibilidad de todas las
familias por estas comprobaciones.
