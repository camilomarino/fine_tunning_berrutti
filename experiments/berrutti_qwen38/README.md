# Qwen3.8-27B LoRA sobre Berrutti v3

Experimento privado de OCR de página completa, independiente del catálogo de
modelos y de los registros publicados del benchmark. Usa la revisión fija
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16, LoRA r16/alpha32 en
el decodificador de lenguaje, encoder visual congelado y dos A100 de 40 GB
particionadas por capas en un solo proceso. Se fija la semilla 20261007.

`split-v1.json` distribuye las 175 páginas como 125/25/25 por rollo. Es un
split **provisional**: agrupar por rollo reduce contaminación, pero aún requiere
auditar identidad documental entre rollos. La prueba mínima entrena sólo dos
páginas y evalúa una de validación y una de test. La corrida completa usa todas
las páginas de cada split durante tres épocas; evalúa validación y test antes
del entrenamiento y al cierre de cada época. El test se muestra como tendencia
descriptiva; se elige el checkpoint por CER de validación. Los resultados
anteriores del benchmark hacen que el test no sea externo ni plenamente ciego.

El CER/WER micro usa la implementación local de `ascii-text-v1` en
`scoring.py`, verificada contra los normalizadores del benchmark.
La inferencia usa `enable_thinking=False` y el mismo presupuesto de imagen en
baseline y checkpoints. Las generaciones que no terminan se registran como
incompletas con las guardas descritas abajo. No se truncan targets: un ejemplo
mayor que 8192 tokens o sin EOS supervisado detiene la corrida antes de aplicar
esa actualización; es un error de datos, distinto de un fallo de generación.

Cada época guarda el checkpoint de adaptador, optimizador, scheduler y RNG
**antes** de evaluar. Las métricas terminadas quedan en `epochs/` y
`evaluations/`, separadas del checkpoint inmutable. En `train.py`, `--resume`
restaura el último checkpoint y termina su evaluación pendiente antes de
continuar entrenando. Si no hubo aún un checkpoint, reinicia desde el modelo
base y recupera los intentos de evaluación ya guardados. La variante curriculum
también guarda antes de evaluar; aún no tiene un CLI de reanudación automática.
Las predicciones,
transcripciones y pesos no se incluyen en Git. `events.jsonl` permite leer el
progreso; los eventos `evaluation` contienen CER/WER y `optimizer_step` registra
pérdida y norma de gradiente. Cada dos horas de actividad el proceso escribe
un resumen en `progress-2h.jsonl`; una sesión de monitoreo puede usarlo para
informar la evolución sin abrir las transcripciones privadas.

`monitor_status.py` recoge por SSH sólo metadatos Slurm y eventos acotados.
`report_to_chat.py` entrega ese resumen al servidor Codex existente mediante
su socket Unix. No usa un segundo `codex exec resume`, porque éste entra en
conflicto con el escritor del chat abierto en la app. Si el chat tiene un turno
activo, sale con código 75 para que el servicio reintente al quedar libre.
El servicio debe reintentar ese código y conservar el recibo de entrega entre
ejecuciones; un envío aceptado se comprueba antes de intentar otro. El intérprete
del servicio necesita `websockets` (la instalación actual usa 16.0).
Esto permite informes en el chat, pero no crea una tarea en la vista nativa
Scheduled. El temporizador y los recibos permanecen privados fuera de Git.

La convolución de los parches visuales se reemplaza por una operación lineal
algebraicamente equivalente tras comprobar condiciones de kernel/stride,
paridad numérica y paridad de pérdida en la primera muestra real. Esto evita
una ruta Conv3d muy lenta en la versión fijada de Transformers.

El entorno Ruche debe preparar previamente las dependencias fijadas y el
kernel `causal-conv1d` de la revisión declarada en `train.py`. `run.sbatch`
usa el entorno privado de la prueba de viabilidad y requiere `EXPERIMENT_ROOT`
y `MODE=sanity|full`. Por ejemplo, dentro de Ruche, tras transferir código y
dataset a un WORKDIR privado:

```bash
sbatch --test-only --export=ALL,EXPERIMENT_ROOT="$WORKDIR/berrutti-qwen38-training/20261007-r1",MODE=sanity run.sbatch
sbatch --export=ALL,EXPERIMENT_ROOT="$WORKDIR/berrutti-qwen38-training/20261007-r1",MODE=sanity run.sbatch
```

## Guardas de generación y evaluación

`generation_guard.py` es un módulo de política del experimento, sin dependencias
del runtime ni de la orquestación del benchmark. `train.py` lo adapta a los
`StoppingCriteria` de Transformers y `train_curriculum.py` reutiliza ese
evaluador. El experimento ya no importa el paquete `berrutti_bench`.

- Reconoce la lista completa de EOS de `model.generation_config`; sólo usa el
  EOS del tokenizer como alternativa si falta la configuración del modelo.
  No fuerza EOS al alcanzar el presupuesto y una salida vacía tampoco es éxito.
- Primer intento greedy: 4096 tokens y 600 segundos. Detecta una cola de
  bloques idénticos de 1 a 128 tokens, con al menos cuatro repeticiones y
  128 tokens repetidos. Comprueba la repetición cada 16 tokens generados,
  sin incluir el prompt; no prohíbe globalmente n-gramas del documento.
- Como máximo dos intentos por página. El segundo siempre permite 8192 tokens
  para cubrir fallos combinados de longitud y repetición; si detectó repetición,
  además usa `repetition_penalty=1.05`. El segundo intento tiene 900 segundos. El límite
  temporal se comprueba entre pasos: no puede interrumpir un kernel CUDA
  bloqueado. Slurm sigue imponiendo el límite duro de la asignación.
- Cada intento conserva texto bruto acotado, causa, tiempo, tokens y parámetros
  en archivos privados, antes de decidir un reintento. No guarda logits ni
  atenciones. No utiliza la referencia ni el CER para elegir un intento.
- Si ambos intentos fallan, conserva y puntúa la última salida completa tal
  como fue devuelta por el generador, aunque esté marcada `incomplete`. El CER
  incluye **todas** las páginas, puede superar el 100 % y se acompaña de
  `failed_pages`, `retried_pages` y las causas. Ninguna página difícil desaparece
  del denominador. Un checkpoint con validación incompleta no es elegible como
  mejor modelo; el test no interviene en esa selección.
- Una repetición, un límite temporal/de tokens o una salida vacía no abortan
  la época. Los errores de CUDA, memoria, archivos o valores no finitos siguen
  siendo errores explícitos; no se disfrazan de transcripciones fallidas.
  Se conservan clipping de gradientes y comprobaciones de pérdida/gradiente.
- Al reanudar con la misma configuración y código se recuperan los intentos
  guardados sin lanzar un tercer intento ni reemplazar evidencia anterior.

Los valores de detección y la penalización de reintento son heurísticas
conservadoras, no una garantía de mejor CER. Se usan las primitivas oficiales de
[generación de Transformers](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
y su soporte de
[criterios de parada y múltiples EOS](https://huggingface.co/docs/transformers/main/en/internal/generation_utils).
[olmOCR](https://github.com/allenai/olmocr/blob/main/olmocr/pipeline.py) también
acota los reintentos y cambia la decodificación después de fallos; no se copia
su política de fallback con texto externo al modelo.

La política y los hashes del código quedan en `config.json`. Debe ejecutarse
un sanity nuevo con estas guardas antes de otra corrida completa, en una ruta
nueva. Las corridas existentes y su código remoto se conservan. Las
comparaciones entre baseline y curriculum exigen la misma política; no se
atribuye al entrenamiento una diferencia causada por cambiar la decodificación.
En curriculum, la evaluación de `stage=region, epoch=0` ya es posterior al
entrenamiento por regiones; sólo `stage=baseline` es anterior al entrenamiento.

Las pruebas sin GPU están en `tests/test_qwen38_experiment.py`. Dentro del
entorno de entrenamiento y de una asignación Slurm de CPU puede ejecutarse
`python -m experiments.berrutti_qwen38.smoke_generation_guards`; usa pesos
sintéticos pequeños sin descargar modelos. `diagnose_generation.py` reproduce
una página con la decodificación anterior y con guardas en dos A100, conservando
los resultados en un directorio nuevo. No actualiza pesos ni ajusta hiperparámetros
según el test.

No registrar ni publicar el adaptador desde este script: antes de su primer
benchmark completo debe pasar `model check`, `model probe` y `model verify`
según `AGENTS.md`.

## Variante de generalización, independiente de la primera corrida

La corrida `full` anterior queda inmutable y **no** usa aumentos ni recortes
de línea. La variante `train_curriculum.py` comienza con hasta ocho regiones
alineadas por página de entrenamiento (989 presentaciones de 124 páginas),
filtradas por longitud y coincidencia normalizada con la referencia de página.
Luego ajusta tres épocas sobre páginas completas. En las presentaciones de
entrenamiento usa pequeñas perturbaciones de contraste y brillo; desenfoque
leve con baja probabilidad; y una rotación menor a 0,4° sólo en páginas, con
expansión y fondo blanco para no cortar texto. Las regiones reciben un margen
de 4 a 12 píxeles antes del procesamiento. Validación y test usan imágenes
originales, sin aumentos, y los mismos parámetros de generación de la primera
corrida. No se recortan páginas completas ni se inventan etiquetas.

La primera fase usa una pasada por regiones y LR 5e-5; la fase de páginas usa
LR 2e-5, AdamW, warm-up del 5 %, decaimiento coseno con piso del 10 %,
acumulación de ocho, clipping de gradientes a 1 y LoRA r16/alpha32 con dropout
0,1 sobre el
decodificador. El encoder visual permanece congelado. El código comprueba la
alineación de cada región, fija semillas y deja una lista privada de cada
presentación y perturbación aplicada. Se evalúan ambas particiones tras cada
fase/época y sólo validación elige el mejor checkpoint. Los resultados de test
son descriptivos: al mirar repetidamente el test durante el desarrollo ya no
es un test ciego.

La rotación poco frecuente y LoRA/AdamW siguen la
[receta oficial de olmOCR](https://github.com/allenai/olmocr/blob/main/olmocr/train/configs/v0.4.0/qwen25_vl_olmocrv4_finetuning.yaml).
Las perturbaciones fotométricas son conservadoras y los recortes de línea
respetan sus etiquetas, como en las
[recetas de reconocimiento de PaddleOCR](https://www.paddleocr.ai/main/en/version2.x/ppocr/model_train/recognition.html).
Ninguna de esas referencias demuestra que esta variante supere al baseline en
Berrutti: ésa es la comparación que medirá la validación.

`build_curriculum.py` genera el manifiesto privado a partir de candidatos
también privados; ambos deben quedar fuera de Git. `run_curriculum.sbatch`
consume ese manifiesto y exige primero un ensayo `CURRICULUM_MODE=sanity` de
cuatro regiones y dos páginas. La corrida `full` de la variante exige un
ensayo completo y la finalización de la primera corrida para poder comparar
contra su CER de validación. Una dependencia Slurm secuencial evita que estas
corridas compitan por las A100 ocupadas por el experimento principal.
