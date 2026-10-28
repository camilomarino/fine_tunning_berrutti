# Estado operativo Qwen–Berrutti

Actualizado el 28 de octubre de 2026. Resume la receta vigente, sin acumular
recetas ni resultados históricos. Ver [THREE_ARMS_125_PROTOCOL.md](THREE_ARMS_125_PROTOCOL.md).

## Organización para el siguiente entrenamiento

El código de trabajo se separó en `data/` y `archive_berrutti/` para datos,
`finetune/recipes/` para recetas, `evaluation/` para métricas y `reporting/`
para auditoría y publicación. Los informes finales están en `reports/berrutti/`.
Ver [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md). Las entradas antiguas
son alias de compatibilidad; los bundles congelados, pesos, datos y la release
publicada no fueron modificados. La reorganización no lanza entrenamientos ni
acredita el runtime de una receta futura.

El siguiente corpus confirmado por el usuario es de **artículos de arXiv**,
separado de la biblioteca Archive Berrutti. Su selección, representación,
targets y evaluación todavía deben definirse; no se descargó ni lanzó.

## Cierre de la auditoría del fine-tuning

La [auditoría final del 22 de octubre](../reports/berrutti/FINETUNING_AUDIT_20261022.md) está cerrada.
Los controles CPU independientes terminaron Slurm COMPLETED/salida cero con
recibos exitosos: 208 checkpoints completos, AdamW/calendario/RNG y 825 casos
únicos de evaluación de página recalculados. No se repitieron updates ni OCR.

- Los tres modelos Berrutti completaron tres épocas y 48 updates por modelo.
  Validación completa elige la época 2 en los tres. Combinado: CER 1,930 %;
  páginas: 2,119 %; líneas: 2,567 %.
- Páginas y combinado tienen test completo: CER 3,141 % y 3,230 %, con cero
  fallidas y tres/cuatro reintentadas. Líneas conserva test provisional,
  24/25 completas, una fallida y dos reintentadas. El caso fallido agotó sus dos
  intentos.
- La extensión LUISA completó sus 64 updates y cuatro épocas. La época 4
  seleccionada tiene CER 2,048 % en validación y 3,220 % en test completo,
  cero fallidas y tres reintentadas. Su propio baseline, el test contaminado
  y la generación diferente impiden atribuirle una ventaja independiente.
- El combinado es la elección por el criterio de validación prefijado.
  Páginas es la alternativa de menor coste observado. La diferencia entre
  ambos no demuestra superioridad general; el informe conserva su incertidumbre.

Los punteros privados y la coordinación registran el cierre y la evidencia.
No hay entrenamiento que reanudar ni pasos confirmados que repetir.
La consulta y el seguimiento continúan sólo manuales; no reactivar timers,
informes, vigilancia ni llamadas automáticas al modelo.

## Viewer de las cuatro variantes

El 26 de octubre el usuario descartó la portada y los visores incrustados de
la release r1. Su [sustitución](FINETUNING_PUBLICATION.md) usa directamente la
copia exacta del OCR Diff Viewer v2 publicado: matriz y explorador completos,
sin portada ni iframe. La release r4 conserva las 50 páginas held-out y 250
predicciones: un único Qwen sin entrenamiento, con la generación determinista
original, y los cuatro ajustes. Retira del visor la segunda evaluación sin
ajuste y las etiquetas «referencia Berrutti/LUISA», a pedido del usuario. Los
nombres indican datos y épocas; se mantienen los IDs de las entradas conservadas.
Está publicada y verificada: árbol remoto completo y manifest idénticos al bundle,
ocho archivos comprobados por HTTPS y 38 controles del navegador público sin
errores. El puntero `current-ocr-diff-finetuning` registra `ACTIVE_VERIFIED`.
Las releases anteriores conservan su evidencia. No modificar el v2 de referencia,
que permanece congelado y conserva su hash público original. La release r5
siguiente conserva otro snapshot y destino independiente.
URL: <https://iie.fing.edu.uy/~cmarino/ocr/qwen_berrutti_comparison/>.

La release r5 está publicada y verificada en Ampere, con resúmenes breves de
las cuatro recetas y la evidencia cerrada consultada el 16 de octubre.
La extensión muestra su baseline propia, 25/64 actualizaciones, época 2 de 4
y 17.198/29.773 imágenes confirmadas en esa época. Conserva el test incompleto
de líneas y el carácter exploratorio de LUISA; los controles intermedios no
seleccionan checkpoint. Esa release conserva el snapshot del 16 de octubre y no incorpora el cierre del 22;
la comparación final se publica por separado en el destino indicado arriba. Es estática y manual, sin reactivar
vigilancia ni informes. Ver [RESULTS_VIEWER_RELEASE.md](RESULTS_VIEWER_RELEASE.md).

## Seguimiento manual por pedido del usuario

El 14 de octubre el usuario pidió retirar todas las llamadas automáticas al
modelo para consultar el entrenamiento, tanto en este chat como en el otro chat
del mismo nombre y repositorio. Se detuvieron, deshabilitaron y retiraron las
unidades de usuario de `berrutti-training-verification`,
`berrutti-ruche-chat-report` y `berrutti-ruche-monitor`; el watcher tiene
`enabled=false`. No hay crontab de usuario ni otra programación del repositorio
en los cron y servicios del sistema revisados. El seguimiento queda sólo manual.
No reinstalar ni reactivar esas llamadas sin una nueva petición explícita.
Esta decisión sustituye todas las instrucciones históricas de mantener activa
la vigilancia o los informes periódicos. Los entrenamientos Slurm, sus
continuaciones nativas, datos, entornos y resultados permanecen intactos.

## Antecedentes técnicos y receta

Las siguientes secciones conservan las comprobaciones y estados fechados durante
la ejecución. Sus menciones a cola, RUNNING, gates pendientes y vigilancia activa
son antecedentes, no el estado actual. El cierre anterior y la política de
seguimiento manual los sustituyen. No reenviar los entrenamientos ni reactivar
automatizaciones basándose en esos estados históricos.

### Comparación y lanzamiento original

- Sólo páginas: las 125 imágenes completas de Berrutti en cada época.
- Sólo líneas: sus 5.212 recortes de una línea en cada época.
- Combinado: una página completa, todas sus líneas y la siguiente página;
  recorre las 125 completas y los 5.212 recortes en cada época.
- Tres épocas completas, 16 actualizaciones por época y 48 por modelo.
- Mismo Qwen fijado e idéntica inicialización LoRA nueva, sin pesos entrenados heredados.
- Validación y test: 25 páginas completas cada uno, idénticas para los tres brazos.
- LUISA fuera de estos tres brazos.

El usuario autorizó el lanzamiento. Los tres trabajos están enviados a Slurm
con dependencias: comprobación CPU de entradas/configuración → ensayo GPU
representativo → baseline compartido → tres brazos.
Consultar el snapshot para su estado real; PENDING es cola o dependencia,
no entrenamiento completado. La configuración privada de coordinación contiene
los trabajos vigentes; no reenviar jobs en cola ni modificar sus fuentes congeladas.
La operación privada se identifica en `.private/three-arms-launch-*/submission.json`.

`runner.py` y `run_three_arms.sbatch` implementan la receta. El ensayo utiliza dos
páginas de origen, una con discrepancias de anotación y otra densa, todas sus
líneas, una actualización por estrategia y generación de una página de cada
split por estrategia. Comprueba dos A100, pérdidas/gradientes finitos y recarga
exacta de adaptador, optimizador, calendario y RNG. Sólo su Slurm COMPLETED y
recibo exitoso habilitan baseline. El CER del ensayo no representa los splits.

Baseline mide 25 páginas de validación y 25 de test con cero actualizaciones.
Se comparte sólo cuando coincide la identidad de modelo, adaptador, fuentes,
entradas, processor y política. Cada brazo evalúa las 25 páginas de validación
tras cada época; elige el menor CER completo entre 1, 2 y 3 y evalúa las 25 de
test sólo para ese checkpoint. Total: 350 evaluaciones de página sin reintentos,
frente a 500 con test tras cada época. Test no selecciona modelos.

Las 125 páginas se mantienen. Hay discrepancias entre las referencias de página
y líneas en 34; se registran sin corregir etiquetas ni descartar páginas.
Una diferencia de CER no se atribuye exclusivamente a la representación visual.

## Runtime, checkpoints y límites

Perfil común: BF16, SDPA y proyección visual FP32, sin caché de embeddings visuales.
La caché de crops verificada conserva píxeles/etiquetas exactos: 5.212 recortes,
140 MB, construida bajo Slurm en 14,6 s con ocho CPU. El runner la consume y
reutiliza en RAM CPU las entradas ya procesadas durante las épocas siguientes.

Las mismas librerías se copian al SSD del job. El diagnóstico CPU terminó con
Slurm COMPLETED: importación en 51 s frente a 505 s desde GPFS; copia más prueba,
2 min 18 s. El ensayo GPU del lanzamiento valida ese perfil completo. No usar
estas cifras como estimación de duración total ni de velocidad de generación.
La caché visual no superó identidad exacta y queda excluida. Los diagnósticos
se conservan aparte en `performance_checks`; no son experimentos adicionales.
No se instaló el kernel `causal_conv1d` identificado ni se mutó ningún entorno activo.

Cupo efectivo observado: cuatro A100, por tanto dos brazos de dos GPU a la vez,
sujeto a cola. Cada segmento solicita ocho CPU, 96 GB, 30 GB de SSD y 24 horas.
Tras 21 horas cede en un límite de grupo/página y envía continuación, máximo
ocho segmentos. Cada actualización confirma checkpoint nuevo con hash y cursor;
restaura adaptador, AdamW, calendario y RNG. El monitor sigue `continuation.json`.
No repetir actualizaciones confirmadas ni páginas con recibo cerrado.

La revisión de cola del 9 de octubre confirmó que todas las A100 operativas
estaban asignadas y el ensayo esperaba por `Resources`; baseline y brazos,
por `Dependency`. Cuenta, partición, CPU, RAM y SSD son válidos, sin nodo
forzado ni reserva exclusiva. Pruebas `sbatch --test-only` con menos RAM,
sin SSD y menor walltime no adelantaron el inicio estimado. No reenviar ni
cancelar trabajos para intentar mejorar prioridad: se conserva su antigüedad.
La cuenta sólo autoriza QoS normal, sin incremento de prioridad de usuario.
Tres brazos simultáneos requieren seis A100; el límite vigente es cuatro en
total, también al usar máquinas distintas. La solicitud de ampliación temporal
y consejo de prioridad a soporte está preparada en privado, sin enviar.
No hay dependencias entre brazos. El monitor añade motivo de cola y una hora
estimada UTC, variable y nunca garantizada; los informes la convierten a París.

## Preparación interoperable

El usuario pidió generalizar modelos y destinos Ruche/ClusterUY. `python -m
finetune` prepara un único wrapper Slurm a partir de perfil y receta privados;
reutiliza TRL SFT para nuevas familias o admite comandos de recetas propias.
Ver [finetune/README.md](../finetune/README.md) para alcance y contrato. No importa
Torch ni envía jobs. El runner Qwen vigente y sus fuentes congeladas no cambian.
El 9 de octubre se configuró y verificó SSH a ClusterUY, incluido su puerto de
transferencia. Se comprobaron asociaciones, QoS, almacenamiento y ocupación
real. Slurm COMPLETED y recibos exitosos acreditan contenedor Python 3.12 y
CUDA host/contenedor en una A100. No son entrenamiento Berrutti ni prueba del
modelo completo. Los dos nodos A40 tienen dos GPU cada uno, actualmente ocupadas;
las A100 son individuales por nodo. No se solicitan dos A100 locales allí.

La transferencia directa entre clusters mostró pérdida y rendimiento insuficiente;
se retiró sólo ese trabajo CPU y se revocó su clave temporal restringida. La copia
usa el acceso existente de Ampere, ocho flujos y un archivo comprimido sin cambios
en las librerías. Fragmentación, reconstrucción y hashes se hacen bajo Slurm;
el ensayo GPU depende de verificar el archivo completo y los datos. Ampere sólo
transporta archivos, sin entrenamiento ni copias de claves privadas.

La operación privada `.private/clusteruy-setup-*/submission.json` registra la
preparación CPU de pesos fijados, transferencia de librerías/datos, ensayo
sintético de modelos y ensayo representativo en dos A40, con dependencias.
El ensayo representativo conserva las dos páginas, todas sus líneas, las tres
estrategias, checkpoint/restauración y generación completa. El launcher común
admite Singularity y sus continuaciones; el runner exige el perfil explícito
`gpu_type: "A40"` y liga la identidad del runtime al hardware. Los trabajos de
Ruche conservan sus fuentes y entornos originales. La ruta de ClusterUY mantiene
CUDA inicializado en un proceso host propio mientras corre Singularity, como
workaround acotado; esto no demuestra una corrección administrativa del driver.

El usuario autorizó retirar de la cola Ruche sólo después de comprobar ClusterUY.
No cancelar por acceso SSH, visibilidad CUDA o éxito del ensayo sintético: exigir
Slurm COMPLETED y recibo exitoso del ensayo representativo completo. Si Ruche
empieza antes, conservar el trabajo en curso y revisar los duplicados pendientes.
Los entrenamientos completos de ClusterUY quedan preparados para esa decisión,
sin sustituir todavía los tres trabajos de Ruche. El primer ensayo Ruche recibió
dos A100, pasó CUDA/routing y cargó el modelo, pero falló por OOM de Triton en el
primer backward de páginas; Slurm canceló sus cuatro dependientes. Se conserva
esa evidencia. Un intento nuevo libera las salidas sin uso antes del backward
y vacía la caché libre del allocator; añade memoria por dispositivo al diagnóstico.
Mantiene pérdida, píxeles, datos y pesos. El ensayo nuevo aprobó el perfil en las
dos páginas representativas, sin reproducir aquel OOM. Ensayo, baseline y tres brazos volvieron a enviarse con fuentes
congeladas nuevas; ClusterUY tiene el mismo candidato en un destino privado nuevo.
El monitor elige el alias SSH por el campo `cluster` de la coordinación (`Ruche` o `ClusterUY`),
sin mezclar los espacios de identificadores Slurm. Las comprobaciones pueden
indicar su propio destino: el seguimiento conserva los tres brazos Ruche e
incluye los ensayos ClusterUY por separado. Un fallo SSH de ese destino se
registra como UNKNOWN, sin ocultar el estado del otro cluster ni publicar el
mensaje privado de conexión. `make verify` pasó 197 tests,
compilación, lint y formato. El ensayo GPU representativo Ruche ya terminó con
Slurm COMPLETED y recibo exitoso; ClusterUY todavía espera recursos/dependencias.

La verificación de pesos comparó los 18 shards con los hashes LFS y tamaños del
modelo fijado de Ruche y detectó una descarga truncada. Un job CPU nuevo terminó
con Slurm COMPLETED: recuperó sólo ese shard, verificó tamaño/SHA256 de los 18
archivos y conservó la copia rechazada. La reconstrucción del archivo de librerías
también terminó con hash idéntico. Otro job CPU terminó con recibo exitoso:
importó Torch, Transformers, PEFT y Accelerate y verificó las 125/25/25 páginas
y los 5.212 recortes. Ninguna de esas comprobaciones acredita todavía backward
ni generación GPU. El ensayo sintético GPU terminó con Slurm COMPLETED y recibo
exitoso para modelos pequeños Qwen y Gemma: forward/backward finitos, una
actualización, checkpoint/restauración exacta y generación. No acredita el
Qwen completo de 27B en ClusterUY. El ensayo ClusterUY depende de la
identidad completa corregida, además del runtime y GPU.
`memory-candidate.json` y `remaining-launch.json` de la operación privada apuntan
al candidato actual; `submission.json` inicial conserva los primeros intentos.

La última cancelación automática de baseline y brazos tuvo otra causa: el
ensayo rechazó en `setup` el perfil nuevo de liberación de salidas porque el
runner todavía aceptaba sólo el nombre anterior. No llegó a cargar el modelo.
`runner.RUNTIME_PROFILE` fija ahora el perfil aceptado; `verify_inputs` valida
perfil, fuentes, cohorte, splits y recibo cerrado de crops antes de crear el
runtime. `--mode preflight` ejecuta esas mismas comprobaciones sin GPU ni
updates. Nuevos preflights CPU terminaron correctamente en ambos clusters,
con 125 páginas de origen y 5.212 recortes. Los intentos rechazados conservan
su evidencia y los nuevos tienen copias congeladas independientes. El nuevo
ensayo Ruche terminó con Slurm COMPLETED y salida cero: dos A100, Qwen completo,
una actualización finita por estrategia, restauración exacta y seis generaciones
completas (una página por split y estrategia), sin páginas fallidas ni reintentos.
El recibo privado de verificación conserva estados Slurm, eventos y cierre del
checkpoint. Son comprobaciones técnicas, sin resultados de la comparación.
La evaluación inicial compartida terminó con Slurm COMPLETED, salida cero y
recibos cerrados coincidentes con el runtime aprobado. Completó las 25 páginas
de validación y las 25 de test con cero updates y ninguna página fallida;
validación sin reintentos, test con uno. La evidencia y los CER quedan privados.
Líneas terminó en Ruche: confirmó las 48 actualizaciones, las tres épocas
completas de 5.212 recortes y las tres validaciones de 25 páginas sin fallos ni
reintentos. La continuación automática restauró el paso 47 y confirmó sólo el
48; el cierre del primer segmento no era el cierre del modelo. La validación
completa seleccionó la época dos. El test conserva 25 recibos, pero sólo 24
generaciones completas: una página agotó sus dos intentos por repetición.
Otra alcanzó el límite de tiempo en el primer intento y completó el segundo;
hubo dos páginas reintentadas en total.
Slurm COMPLETED y el recibo de cierre acreditan la ejecución, no un test
completo; su CER sigue siendo provisional. No repetir actualizaciones, borrar
recibos ni efectuar un tercer intento bajo la política vigente. El combinado
falló durante el backward de una página completa en la primera época: solicitó
2,95 GiB en una A100 de 40 GB con 2,50 GiB libres. Su checkpoint de origen conserva nueve actualizaciones
confirmadas, 72 páginas de origen y 3.046 imágenes presentadas. El grupo parcial
no actualizó pesos. Su recuperación privada `combined-recovery-*` conserva todas
las fuentes matemáticas y restaura ese checkpoint con AdamW, calendario y RNG;
sólo prueba el allocator `expandable_segments:True` en un proceso nuevo. Se
enviaron un preflight CPU, un segmento de recuperación y la continuación con
dependencias. El preflight terminó con Slurm COMPLETED, salida cero y recibo
exitoso: hashes y CRC del checkpoint/cursor, fuentes y cohorte completa. El
segmento GPU terminó con Slurm COMPLETED, salida cero y recibo exitoso. Confirmó
exactamente la actualización diez: ocho páginas completas y todas sus líneas,
llegando a 80 páginas de origen y 3.642 imágenes presentadas. Verificó recarga
exacta y generación completa aparte, y publicó `continuation.json`. Su
continuación ya enviada obtuvo el cupo de GPU al terminar páginas y está RUNNING
en la segunda época, tras cerrar la primera y su validación completa;
conserva el checkpoint diez y el allocator verificado en su wrapper congelado.
No repetir esa actualización.
`combined_memory_recovery` tiene el mismo contrato de cierre que la recuperación
de páginas, no constituye otro modelo ni el ensayo de las tres estrategias.
La vigilancia incluye su cierre/fallo y el del preflight. El cierre acredita
la recuperación de ese grupo, no las épocas restantes. Los trabajos activos o pendientes de los otros brazos
conservan fuentes, recursos y dependencias originales.
Páginas falló por OOM durante el backward de la siguiente página tras 76 ejemplos
procesados; tiene nueve actualizaciones confirmadas y 72 páginas en su checkpoint.
El grupo parcial no confirmó ninguna actualización adicional. La evidencia del
fallo se conserva y la recuperación usa ese checkpoint, AdamW, calendario y RNG,
sin repetir las nueve actualizaciones. Su primer segmento de recuperación ya
terminó correctamente; la continuación retomó desde el paso diez y ya terminó
con Slurm COMPLETED y salida cero. Cerró las tres épocas, 48 actualizaciones,
las tres validaciones completas y el test completo del checkpoint elegido por
validación. No hay páginas fallidas; el test conserva sus reintentos. El estado
con época cuatro es el cursor posterior al bucle de tres épocas, no una cuarta
época entrenada. Los recibos y los resultados permanecen privados.
Todos parten de la misma identidad de runtime e inicialización LoRA aprobadas.
Controles CPU independientes terminaron con Slurm COMPLETED y recibos exitosos:
los primeros checkpoints de páginas y líneas coinciden con sus SHA256 y sus
archivos Torch superan el control CRC. No cargaron el modelo ni efectuaron
actualizaciones; la restauración completa ya quedó acreditada por el ensayo
GPU representativo.
La recuperación de páginas está sellada en una operación privada nueva
`.private/pages-recovery-*/recovery-definition.json`. Conserva byte por byte las
fuentes matemáticas, entradas, píxeles, pesos y política de generación; prueba
`PYTORCH_ALLOC_CONF=expandable_segments:True` en un proceso nuevo. La memoria
reservada sin usar hace razonable esa candidata, pero no demuestra que resuelva
el fallo. Un preflight CPU valida los hashes del checkpoint, cursor y controles.
El primer segmento GPU restaura el paso nueve y confirma sólo el grupo siguiente
completo: paso diez y 80 páginas acumuladas. Verifica recarga y una generación
completa en una carpeta separada; su CER técnico no selecciona modelo ni ocupa
la validación de la época. La comprobación terminó con Slurm COMPLETED, salida
cero y recibo exitoso: actualización diez confirmada, 80 páginas acumuladas,
recarga exacta del checkpoint y generación completa. La continuación ya enviada
conserva ese paso diez y el mismo allocator, verificado en su comando de envío.
No repetir esa actualización. Ese recibo acredita la recuperación del grupo que
falló; el cierre posterior de la continuación acredita las tres épocas. Es parte del mismo brazo, no
otro modelo ni un ensayo de las tres estrategias.
La coordinación conserva líneas y registra las recuperaciones en sus mismos
brazos, sin cambiar la comparación. El cupo de cuatro A100 limita su concurrencia.
Seguir `continuation.json` y `next-job.json` de esta operación al verificar el
cierre. El monitor requiere actualización confirmada, recarga y generación para
aceptar `memory_recovery_completed`; la vigilancia reactiva el chat al cerrarse
esa comprobación. No reenviar trabajos ya enviados ni editar sus fuentes.
El monitor reconoce carga completada, verificación de proyección y preparación
del backward sin seguir mostrando una fase de carga antigua ni inventar updates.
El intento anterior de ClusterUY falló en el control sintético de gradientes del routing,
antes de cargar Qwen: dos A40 visibles y CUDA operativo, sin actualizaciones
Berrutti. La referencia calculaba la pérdida en GPU 1 mientras Accelerate
devolvía la salida a GPU 0. El candidato nuevo calcula ambas pérdidas en el
mismo dispositivo y conserva la igualdad exacta de salidas/gradientes; añade
el parámetro y la diferencia al error. No dar por confirmada la causa numérica
ni por aprobada la corrección antes de verificarla en GPU. Un diagnóstico corto
contrasta dispositivo de pérdida, checkpointing y determinismo, sin updates ni
generación. Se conservaron el intento fallido y sus fuentes. Nuevos preflight
y ensayo completo están enviados con fuentes y rutas nuevas; los punteros
privados actuales identifican el candidato vigente. `make verify` pasó los
197 tests, compilación, lint y formato. El preflight nuevo terminó con Slurm
COMPLETED, salida cero y recibo exitoso para 125 páginas y 5.212 líneas; el
ensayo completo está PENDING por prioridad, sin dependencias fallidas.
La QoS rápida aceptó el test del
scheduler, pero bloqueó la asignación GPU por su límite efectivo; se restauró
la QoS GPU del mismo diagnóstico, sin reenviarlo. El ensayo completo conserva
la QoS GPU y cuatro horas. Las estimaciones de cola son variables.
La aprobación sigue exigiendo Qwen completo, las tres representaciones,
recarga de checkpoints y generación en dos A40, con Slurm COMPLETED y recibo
exitoso. Ruche conserva la continuación del combinado activa en la segunda época
tras aprobar su recuperación. Líneas cerró entrenamiento y validación; su test
agotó los intentos de una página y queda incompleto. Páginas terminó con sus tres épocas,
selección por validación completa y test descriptivo del checkpoint elegido,
sin alterar fuentes ni checkpoints previos. La vigilancia automática de aquel
seguimiento fue retirada por el usuario el 14 de octubre. La coordinación privada distingue ejecución verificada de resultado
completo y conserva el diagnóstico de los intentos agotados.

El seguimiento anterior usaba un timer de cinco minutos y el cliente
`report_to_chat.py --prompt-file`, con recibos privados por evento. El usuario
retiró ese mecanismo y los informes periódicos el 14 de octubre. Su configuración
privada queda deshabilitada; las unidades systemd se retiraron y se conservaron
copias de auditoría. El código de consulta puede usarse manualmente cuando el
usuario pida el estado, sin programar llamadas al modelo.

Máximo dos intentos por página, contando intentos interrumpidos ya reservados.
CER con páginas incompletas es provisional y no selecciona checkpoint.
No ocultar fallos CUDA, pérdidas o gradientes no finitos.

La vigilancia local conserva un registro privado de avisos COMPLETED ya
revisados, ligado al trabajo exacto, recibo exitoso y huellas de su evidencia
independiente. Evita reactivar controles ya auditados dentro de un cierre
posterior. El registro no suprime trabajos o estados distintos, recibos
inválidos, archivos ausentes ni huellas diferentes. Sólo afecta
notificaciones: no acredita optimizaciones ni cambia gates, fuentes Slurm,
cadencia, destino o contenido del informe periódico.

Los tests CPU de la receta verifican
cobertura completa, pesos por página, calendario y recuperación de generación.
El ensayo GPU usa las fuentes congeladas del lanzamiento, sin tocar diagnósticos activos.

La revisión actual del ensayo ClusterUY confirmó Slurm FAILED, salida uno,
en el control de gradientes del routing, antes de cargar Qwen y sin updates
Berrutti ni generaciones. CUDA estaba operativo con dos A40. El diagnóstico
sintético cerrado conservó diez filas con discrepancias y valores no finitos,
incluso en repeticiones de routing explícito: Slurm COMPLETED sólo acredita
su ejecución, no equivalencia ni una optimización aceptada. La evidencia
actual está preservada y no justifica reenviar un ensayo idéntico.

El control privado `a40-transfer-routing-isolation-v1` cerró con Slurm
COMPLETED, salida cero y recibo exitoso; se revisaron 36 filas de transferencias,
cuatro de cálculo y diez de routing. En la asignación A40 observada, las copias
directas y sincronizadas de GPU cero a GPU uno corrompen valores, incluidos
enteros; el cálculo íntegro dentro de cada GPU sí conserva salidas y gradientes
exactos. Pasando por RAM, todas las copias y ambos controles de routing tienen
identidad exacta y finitud. No identifica una causa del driver o hardware ni
acredita Qwen completo, generación o gate. Evidencia cerrada en el puntero
`.private/current-clusteruy-routing-diagnostic.json`; no reenviar ese control.

La corrección fundada usa una candidata y fuentes congeladas nuevas:
`cuda_transport.py` conserva los overloads de `Tensor.to`, desvía únicamente
copias entre dispositivos CUDA por RAM y preserva su cadena de autograd.
Comprueba bytes, fuente, dtype y forma y restaura la interfaz incluso ante
excepción. No cambia pesos, cómputo, datos, píxeles, referencias, calendario,
semilla ni generación. El runner y su control exacto original de 128 muestras
permanecen idénticos. Un preflight nuevo y el ensayo representativo íntegro en
dos A40 deben comprobar las tres estrategias, recarga y generaciones completas;
el recibo de transporte sólo complementa ese gate. No repetir los tres modelos
Ruche ya entrenados ni añadir intentos al test incompleto de líneas. La candidata
de prefill PyTorch del experimento cuatro en Ruche conserva sus propias fuentes,
recibos y contrato; su cierre fallido se describe debajo. La vigilancia permanece
habilitada y conserva su timer activo y los informes periódicos intactos.

La candidata de transporte pasó el control original de 128 muestras y cargó
Qwen completo, pero terminó Slurm FAILED, salida uno, al iniciar la primera
representación: Triton no encontró un compilador C. No confirmó actualizaciones
ni generaciones y no acredita el gate. La corrección de entorno se prepara en
una raíz nueva: añade únicamente archivos del compilador y headers al contenedor
copiado y exige identidad de sus archivos de ejecución, incluida la misma
versión y binarios Python. Una auditoría independiente del SquashFS confirmó
5.916 archivos originales idénticos y dos cambios de metadatos de empaquetado
(receta y etiquetas Singularity), registrados con hashes exactos. El preflight
extrae el SquashFS directamente: convertirlo primero con `singularity build`
regenera esos metadatos y producía un rechazo del propio verificador. Comprueba
compilación/carga C antes del preflight y
compilación del driver Triton dentro del nuevo ensayo GPU. Conserva el bundle de
librerías, las fuentes matemáticas, transporte por RAM y todos los controles del
ensayo íntegro. No modificar ni reenviar la candidata fallida.

El verificador obtiene el descriptor de filesystem de la cabecera SIF y lo
contrasta con la auditoría cerrada. Un control CPU posterior aisló otro fallo
del verificador: Python del host y del contenedor decodifican de forma distinta
dos nombres con acentos y un enlace, aunque sus bytes coinciden. La copia nueva
compara bytes exactos de nombres y destinos de enlaces y conserva los hashes,
permisos e inventario completo. Ocho controles locales verifican esa equivalencia
y rechazan cambios de contenido, permisos, cobertura, enlace y alias de bytes.
Los preflights rechazados y sus dependientes cancelados automáticamente antes
de ejecutarse quedan preservados; no acreditan ni refutan el ensayo GPU.

El ensayo vigente de compilador y transporte por RAM cerró Slurm COMPLETED,
salida cero, con recibos nativo, transporte y compilador exitosos. El validador
local congelado aprobó Qwen completo en dos A40: tres actualizaciones
diagnósticas finitas, recarga y restauración de checkpoints y seis generaciones
completas, sin páginas fallidas ni reintentos. Conserva el routing original
exacto de 128 muestras, doce controles exactos de transporte y la inicialización
común. Su preflight verificó las 125 páginas, 5.212 líneas, 34 fuentes y 8.238
archivos del contenedor. Queda acreditada la compatibilidad de este ensayo;
no se extrapolan CER o velocidad ni se habilita la extensión de Ruche.
No hay modelos originales pendientes de migrar: la lista de lanzamientos sigue
vacía y no se envían duplicados ni otro intento del test incompleto de líneas.
Aquel seguimiento quedó retirado el 14 de octubre; la extensión también está cerrada y auditada.

## Experimento cuatro autorizado

Estado verificado el 13 de octubre: la candidata con el muestreo oficial
autorizado pasó el preflight completo y el ensayo Qwen en dos A100, ambos con
Slurm COMPLETED, salida cero y recibos exitosos. El verificador LOCAL congelado
aprobó evidencia fresca: tres updates diagnósticos finitos (mixto, mixto,
consolidación), recarga exacta de adaptador/AdamW/calendario/RNG, cinco probes
finales hasta EOS y seis controles de muestreo. Todos los probes terminaron al
primer intento; se conservaron los contratos numéricos anteriores. La misma
composición fija de dos columnas terminó también antes de entrenar. Las 36
fuentes congeladas coinciden localmente y en remoto; el preflight verificó las
125 páginas, 5.212 líneas Berrutti, 23.774 LUISA, 1.991 composiciones y los
25.765 hashes completos con cuatro lectores.

Se ejecutaron la recuperación de recibos en modo real y el helper idempotente
de esta candidata. La línea base propia cerró Slurm COMPLETED, salida cero y
recibo exitoso: 25/25 páginas en validación y test, CER 2,904 % y 5,538 %,
cero fallidas y cero/tres reintentadas, respectivamente. Una comprobación
independiente recalculó las 50 páginas desde referencias selladas y predicciones
cerradas; confirmó 53 intentos totales, semillas y contratos de generación
intactos, máximo dos intentos y cero updates. El trabajo de entrenamiento de 64
updates está RUNNING tras satisfacer `afterok`; su nueva asignación de dos A100
pasó el control CUDA/NVML/kernels. Qwen completo cargó con la misma inicialización
y runtime del baseline; verificó otra vez los 25.765 hashes con cuatro lectores.
El 14 de octubre hay ocho actualizaciones reales contiguas con pérdida y norma
de gradiente finitas: 64 páginas de origen y 15.239 imágenes de las cuatro
familias. El primer segmento cerró Slurm COMPLETED, salida cero, después del
paso seis; su continuación nativa está RUNNING y registró los pasos siete y
ocho, sin duplicados. Conserva recursos, inicialización y runtime; la nueva
asignación de dos A100 y los 25.765 hashes con cuatro lectores pasaron otra vez.
Los controles intermedios tras los pasos cuatro y ocho cerraron 25/25 páginas
cada uno, CER 2,992 % y 2,745 %, sin fallidas ni reintentadas. Las comprobaciones
independientes recalcularon las métricas desde referencias selladas y
predicciones cerradas y verificaron los contratos numéricos y de muestreo.
También se cotejaron de nuevo las 36 fuentes congeladas locales y remotas.
Los marcadores 104 y 108 no cierran épocas ni seleccionan checkpoint;
todavía no terminó la primera época. El segundo CER es menor al baseline
propio de 2,904 %, como resultado exploratorio intermedio, sin certificar una
mejora independiente. Cursor y recibo cerrado del checkpoint paso ocho coinciden;
este control local no cargó ni rehasheó independientemente su payload y no
sustituye la auditoría independiente bajo Slurm de los 64 pasos al finalizar.
La coordinación refleja entrenamiento en curso y conserva las pruebas anteriores.
Se verificaron
comentarios únicos, dependencias y recursos contra el lanzador congelado.
El ensayo usa cuatro horas y la producción conserva las 24 horas
del lanzador anterior, con dos A100, ocho CPU, 96G y 30G temporales. Consultar
siempre el puntero vigente, su submission, Slurm y recibos frescos; no enviar
otra cadena ni volver a ejecutar verificadores sobre sus pruebas ya cerradas.
Esta extensión conserva un presupuesto interno de 16 horas por segmento,
comprobado entre grupos, y un máximo de 16 segmentos; esos valores son distintos
de los de los tres brazos originales. Seguir su continuación nativa.
La vigilancia y los informes periódicos se retiraron por pedido del usuario
el 14 de octubre; las comprobaciones posteriores serán manuales.

El cierre del runtime permite ejecutar el entrenamiento; la calidad queda
pendiente de las evaluaciones completas. El probe de línea LUISA llegó a EOS
con una transcripción todavía muy errónea, observación que queda preservada
sin convertirla en una estimación del corpus. Los CER de esta extensión siguen
siendo exploratorios, con contaminación conocida del test. No se reenvían los
tres modelos Berrutti ni se añade otro intento al test incompleto de líneas.

### Antecedentes cerrados del diagnóstico

La primera candidata del muestreo autorizado cerró FAILED, salida uno, durante
la inicialización CUDA/NVML de una asignación real de dos A100, antes de cargar
Qwen o probar la política. Se preservaron Slurm, errores y eventos: cero updates
y generaciones. Esto no rechaza el muestreo ni demuestra causa administrativa
o de hardware. La recuperación usa otra raíz y conserva exactos todos los Python
del modelo y del muestreo. Sólo cambia el lanzador: excluye el nodo de la
asignación fallida y comprueba driver/NVML/CUDA/dos A100 antes de iniciar el
runner. Los recursos son iguales y cada segmento conserva su recibo de asignación
independiente. No sobrescribe el primero al continuar entrenamiento.
La nueva cadena tiene preflight completo y un único GPU `afterok`, 36 archivos
congelados, `make verify` con 243 tests y controles locales del nuevo verificador.
El verificador local congelado y el helper exigen el recibo de asignación del
mismo job además del ensayo íntegro de Qwen con los cinco probes y todos los
contratos previos. Un probe de asignación exitoso no abre el gate. Consultar el
puntero vigente y sus recibos frescos. En ese estado inicial, baseline y los
64 updates reales todavía no estaban enviados.

Al iniciar la corrección de LUISA se comprobó el cierre fallido en Slurm y su
diagnóstico, antes de enviar baseline o entrenamiento. Un diagnóstico independiente
de los controles cerrados observó las cuatro proyecciones de entrada
de la primera capa en las posiciones cero y uno del mismo prefijo cerrado
retokenizado. Usa el runtime fallido exacto, la misma imagen y la inicialización
común; los hooks devuelven `None` y no sustituyen salidas. Conserva cobertura de
64 capas/128 tensores, tres prefills, referencia FP64 acotada con los mismos
pesos y entradas, replay FP32 con umbral 1e-5 y restauración de interfaces/RNG.
El preflight nuevo verifica el corpus completo y las fuentes congeladas; el
diagnóstico GPU depende de su éxito. Consultar
`.private/current-projection-observation.json`, su submission, Slurm y recibos
frescos; exigir también el verificador local congelado. No reenviar controles
cerrados. Esta observación no genera, no entrena y no acredita causa, solución
o gate. El diagnóstico terminó con Slurm COMPLETED, salida cero y recibo nativo
exitoso, pero el verificador local congelado rechazó el supuesto de entradas
ocultas idénticas: ya difieren antes de las proyecciones. Se preservaron las
capturas y el rechazo independiente; no se acredita causalidad ni se reenvía.
Una corrección de runtime todavía exige el ensayo íntegro con los cinco probes
fijos y todos los contratos existentes.

La ficha oficial de la revisión fijada recomienda, en modo instruct, muestreo
con temperatura 0,7, top-p 0,8, top-k 20 y penalización de presencia 1,5. La
receta fallida fuerza greedy. El usuario autorizó explícitamente probar esa
política oficial sólo en el experimento cuatro. Se envió una candidata nueva con
semillas por página/imagen/intento, restauración de RNG/interfaces y los mismos
datos, cinco probes, guard y máximo de dos intentos. Sus controles locales
pasaron; esos controles locales solos no acreditaban Qwen completo. `make verify`
pasó 243 tests y los cinco controles adicionales con PyTorch CPU real pasaron;
el verificador congelado acepta el fixture válido y rechaza 14 inválidos. La
cadena nueva contiene preflight completo y un único GPU dependiente `afterok`,
con 33 fuentes/evidencias congeladas. Consultar `.private/current-experiment4.json`
y su submission; `fetch-evidence.py` y `validate-completion.py` de esa raíz
verifican evidencia fresca, este último localmente (`--preflight` sólo CPU).
Esa primera cadena falló en la inicialización CUDA antes de cargar Qwen;
la recuperación de 36 fuentes y su gate aprobado se describen arriba.
La aprobación
reemplaza el greedy anterior únicamente en esta extensión; conserva intactos los
tres modelos originales. El cambio requiere un preflight completo nuevo y un
ensayo íntegro con cinco probes antes de baseline o entrenamiento. La penalización
de presencia se aplica una vez por token generado distinto, excluyendo el prompt;
la penalización multiplicativa queda en 1,0 en ambos intentos. Las semillas usan
página, hash de imagen e intento y conservan el número de intento al reanudar.
Fuente: [ficha de la revisión fijada](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md).

El ensayo de la candidata con prefill PyTorch terminó FAILED, salida uno, por
repetición de la misma composición de dos columnas en ambos intentos iniciales
y finales. Completó tres actualizaciones diagnósticas finitas con restauración
exacta y cuatro de los cinco probes finales; los controles de precisión pasaron.
No resolvió la repetición ni habilita baseline o los 64 updates reales. Se
preparó un control separado de caché frente a prefijo completo usando ese mismo
runtime fallido, la misma imagen y secuencia retokenizada cerrada, ocho posiciones
y cobertura de 64 capas/128 tensores. Conserva los controles anteriores cerrados,
sin nuevas generaciones, intentos, entrenamiento o CER. Sus diferencias sólo
pueden orientar el diagnóstico; no acreditan causalidad, resolución ni gate.

Ese control guardó las ocho comparaciones y restauró las interfaces, pero cerró
FAILED por un `NameError` propio: `math` sólo estaba importado dentro de otra
función. No produjo recibo exitoso. La evidencia queda preservada y una copia
nueva añade únicamente el import de módulo, sin cambiar runtime, entradas,
posiciones ni umbral de 1e-5. El control local reproduce el fallo anterior y
rechaza 17 controles inválidos; `make verify` pasó 237 tests. Se envió un preflight
completo nuevo y un único diagnóstico GPU dependiente de su éxito. Consultar el
puntero vigente y exigir Slurm COMPLETED, salida cero y recibo completo; esta
corrección del diagnóstico tampoco abre el gate de producción.

La copia corregida cerró Slurm COMPLETED, salida cero y recibo exitoso; el
verificador local congelado confirmó las ocho comparaciones, 64 capas y 128
tensores por rama, finitud, identidad de entradas y restauración. El control
lineal conserva el umbral de 1e-5 y obtuvo 1,91e-7. El token de mayor puntuación
coincide en las ocho posiciones, aunque persisten diferencias de logits de
hasta L2 relativo 0,222. Es ejecución completa del diagnóstico, con cero
entrenamiento, generaciones, intentos o CER; no demuestra equivalencia numérica
ni causa o resolución de la repetición. La candidata de runtime sigue FAILED
y su baseline y entrenamiento permanecen bloqueados. No reenviar el control.

El 10 de octubre el usuario pidió implementar y probar una extensión separada
con todo LUISA, las páginas y líneas train Berrutti, y páginas sintéticas de una
y dos columnas. Esta autorización reemplaza el carácter de propuesta futura
para esa extensión; no altera los tres brazos congelados ni permite reentrenarlos.
Ver `docs/EXPERIMENT_4_LUISA_SYNTHETIC.md`. Tres épocas mixtas y una cuarta de
consolidación, 64 updates, baseline propio y validación completa cada cuatro
updates. Sólo los cierres de época seleccionan checkpoint. La inclusión literal
de las 23.774 líneas LUISA incorpora solapamientos conocidos y el test publicado
LUISA: sus CER se deben identificar como exploratorios, con test contaminado y
sin independencia documental certificada. No ocultar esos límites ni filtrar
datos silenciosamente. La preparación audita todos los originales y registra
la cobertura completa de las composiciones, con targets humanos en orden explícito.

El monitor conserva los tres brazos en `primary_experiments` y sigue la extensión
en `additional_experiments`; su baseline y ensayo son fases, no nuevos modelos.
La vigilancia existente debe seguir hasta verificar también esta solicitud,
avisando CER de validaciones completas, fallos y finalización. No modificar el
timer ni el contrato de sólo lectura de los informes periódicos existentes.

Se congeló una operación nueva y se enviaron preparación/auditoría CPU y ensayo
GPU dependiente con dos A100. No se han enviado aún baseline ni entrenamiento
de la extensión: el helper privado `launch-after-smoke.py` exige cierre Slurm
y recibos completos antes de enviarlos, conserva recibos de envío idempotentes
y actualiza la coordinación. La raíz vigente está en el puntero privado del
experimento cuatro. El ensayo usa el routing y proyección Ruche ya aprobados,
sin incorporar la candidata ClusterUY pendiente. El combinado original sigue
activo y conserva sus GPU. `make verify` pasó 205 tests, compilación, lint y
formato; el nuevo GPU assay sigue siendo necesario y no se acredita por esos
tests CPU.

La primera preparación del experimento cuatro agotó el límite CPU de una hora
durante la síntesis, después de cerrar la auditoría de las 23.774 líneas. No
publicó manifest ni recibo final; el ensayo GPU dependiente se canceló antes de
asignar GPU. Un diagnóstico CPU bajo Slurm confirmó ese punto de corte. Se
conservaron las fuentes y todos los archivos de ese intento. La operación
vigente es un nuevo bundle: reutiliza crops verificados sin pérdida, mantiene
el mismo RNG/layout y referencias, compone con cuatro trabajadores CPU, guarda
avances y recibos por página, y contrasta PNG/targets/procedencia de ambos
layouts contra los originales. La preparación ahora solicita cuatro horas CPU,
sin cambiar la solicitud GPU. Cerró con Slurm COMPLETED, salida cero y recibos
exitosos de preparación y preflight. Verificó todas las líneas individuales y
su cobertura exacta en cada época sintética: 664, 662 y 665 páginas, 1.991 en
total. Las comprobaciones reales de ambos layouts confirmaron identidad exacta
de PNG, targets y procedencia respecto de los originales. Ese ensayo Qwen
terminó FAILED con salida no cero: entrenó las cuatro familias con pérdida y
gradientes finitos y restauró su checkpoint, pero la composición más extensa
de dos columnas repitió en ambos intentos. Las dos páginas reales y la
composición de una columna sí llegaron a EOS; la línea LUISA todavía no se
había probado cuando abortó. No hubo OOM ni recibo exitoso, y no se enviaron
baseline/train. La evidencia, fuentes y resultados de ese intento se conservan.

El 11 de octubre se congeló y envió una candidata nueva. Reutiliza sólo lectura
la preparación completa, sin modificar composiciones, referencias o píxeles;
un nuevo preflight bajo Slurm verifica nuevamente todos los hashes. No reenvía
la preparación ya cerrada. El ensayo nuevo depende `afterok` del preflight
vigente. El job histórico de preparación se comprueba en contabilidad como
`data_prepare`; no se usa como dependencia del controlador una vez retirado de
su memoria. La coordinación apunta a los eventos del intento nuevo, no al
diagnóstico anterior. El preflight nuevo cerró con Slurm COMPLETED, salida cero
y recibo exitoso: verificó los hashes, todas las líneas individuales y la
cobertura exacta por época de las 1.991 composiciones. Su contrato conserva
exactamente la página de dos columnas que falló anteriormente. El ensayo nuevo
terminó FAILED: la primera actualización y su restauración exacta cerraron,
pero el segundo backward agotó memoria GPU. No tiene recibo de aprobación.
Baseline y entrenamiento de la extensión siguen sin enviar.

La candidata conserva los cinco probes originales, incluida la misma página
de dos columnas fallida. Añade un control previo de esa página con la
inicialización original y dos updates mixtos más uno de consolidación, con
checkpoint/restauración tras cada update. Contrasta todos los estados AdamW,
calendario y RNG; guarda los resultados de las cinco generaciones finales
aunque alguna falle. Son tres actualizaciones diagnósticas, no épocas ni
actualizaciones del modelo definitivo. Se mantienen los límites de generación
y dos intentos por página/checkpoint. No se afirma haber resuelto la repetición:
el gate sigue exigiendo Slurm COMPLETED, salida cero, tres updates finitos y
restaurados y cinco generaciones finales completas del Qwen en dos A100. El
helper local vigente compara su contrato con el preflight antes de enviar
idempotentemente baseline/train. No reenviar una cadena ya enviada.

El fallo vigente añade evidencia distinta: la composición difícil repite también
con los pesos iniciales, y el segundo backward solicita 2,95 GiB con sólo
1,57 GiB libres después de inicializar/restaurar AdamW. La nueva candidata
mantiene los momentos AdamW en RAM durante backward y los devuelve sin conversión
a la GPU correspondiente antes de cada paso; los contadores no se trasladan.
No mueve pesos del modelo ni cambia datos, píxeles, etiquetas, pérdida, calendario,
RNG, probes, generación o recursos. El ensayo contrasta cuatro pasos de AdamW
contra una referencia residente en cada GPU, además de las tres actualizaciones
Qwen y sus restauraciones exactas. Registra memoria antes/después y requiere
los momentos en RAM tras restaurar. Sigue sin acreditar resolución de generación
ni habilitar baseline hasta cerrar el gate completo.

La cadena nueva ya se envió con fuentes congeladas y rutas nuevas. Su preflight
cerró con Slurm COMPLETED, salida cero y recibo exitoso: verificó otra vez los
hashes, las 23.774 líneas LUISA y las 125 páginas/5.212 líneas Berrutti, con
cobertura completa de las 1.991 composiciones. Su contrato conserva los cinco
probes y exige el control nuevo de identidad AdamW. El ensayo GPU cumplió la
dependencia `afterok` y sigue PENDING por prioridad; todavía no tiene recibo GPU.
La preparación original cerrada se comprueba en contabilidad y no se reenvía;
los hashes se vuelven a verificar. Baseline/train todavía no están enviados.
`make verify` pasó 210 tests, compilación, lint y formato. El helper nuevo importa
el validador de su copia congelada y exige también el control de identidad AdamW.

La vigilancia de cinco minutos sigue habilitada y el combinado original continúa
activo. Cerró la validación de la segunda época con 25 generaciones completas,
sin páginas fallidas ni reintentos, y empezó la tercera época desde sus 32
actualizaciones confirmadas, sin repetirlas. No cambian sus fuentes ni recursos.
El cierre de preparación o un control inicial exitoso no acredita
aprobación del ensayo nuevo ni entrenamiento de la extensión. Consultar siempre
el puntero privado vigente, los recibos y el estado Slurm real.

La comprobación operativa posterior confirmó FAILED, salida no cero y ausencia
de recibo exitoso del ensayo de memoria. Completó sus tres actualizaciones
diagnósticas finitas, con restauración exacta de adaptador, AdamW, calendario
y RNG, y momentos en RAM. No hubo OOM. De las cinco generaciones finales,
cuatro llegaron a EOS; la misma composición de dos columnas repitió en ambos
intentos. El control inicial también repitió. No son épocas ni resultados del
experimento y no habilitan baseline/train.

La siguiente candidata investiga la precisión de la proyección de salida:
scores FP32 durante generación, conservando exactamente los pesos almacenados
BF16, el entrenamiento y su inicialización. El nuevo contrato exige controles
de copia exacta de pesos y referencia FP32 en los seis decodes (control inicial
y cinco probes finales); registra empates BF16 y cambios de argmax a intervalos
fijos, sin seleccionar por referencia ni agregar intentos. Restaura el forward
original al salir, incluido un fallo. Esta hipótesis no acredita la causa ni
la resolución de la repetición: siguen siendo obligatorios Slurm COMPLETED,
salida cero y el gate completo. El corpus sellado y los tres brazos originales
se conservan intactos.

La cadena de precisión ya se envió en una raíz nueva con fuentes congeladas.
Su preflight cerró con Slurm COMPLETED, salida cero y recibo exitoso: verificó
125 páginas y 5.212 líneas Berrutti, 23.774 líneas LUISA y cobertura completa
por época de las 1.991 composiciones. Su contrato coincide con la copia local
congelada, incluidos los mismos cinco probes y los controles de precisión.
El ensayo cumplió la dependencia `afterok` y sigue PENDING por Resources,
con dos A100 solicitadas; todavía no comenzó ni tiene recibo GPU. Baseline y
entrenamiento siguen sin enviar. No reenviar la cadena en cola. La coordinación
conserva la evidencia nueva y apunta a esta candidata. El combinado original
sigue RUNNING, con 42 de sus 48 actualizaciones confirmadas y la tercera época
en curso. La vigilancia de cinco minutos continúa activa y habilitada;
el informe periódico conserva su timer.

El ensayo de proyección FP32 terminó FAILED, salida no cero y sin recibo de
aprobación. La misma página de dos columnas repitió los dos intentos antes y
después de los updates. Los seis controles de proyección pasaron; hubo tres
updates finitos, restauración exacta de adaptador/AdamW/calendario/RNG y cuatro
probes finales completos, sin OOM. La proyección FP32 no resolvió el fallo.
Baseline y entrenamiento de la extensión siguen sin enviar. La siguiente
candidata conserva esos controles y calcula SDPA matemático FP32 sólo en la
atención completa del decodificador durante inferencia. Contrasta filas acotadas
con FP64, conserva las máscaras/escala y los pesos/cachés almacenados y restaura
la interfaz original al salir. La causa de la repetición sigue sin demostrarse;
esta hipótesis requiere un ensayo nuevo completo en dos A100, sin cambiar datos,
píxeles, referencias, entrenamiento, semilla, calendario, recursos ni generación.
La cadena nueva ya se congeló y envió en rutas nuevas. El preflight está RUNNING
y el ensayo GPU espera `afterok` de ese preflight; baseline/train siguen sin
enviar. El helper importa el validador de su propia copia congelada y exige seis
controles de atención con filas FP64, además de los seis controles de proyección
y del contrato anterior completo. `make verify` pasó 212 tests, compilación,
lint y formato. La vigilancia de cinco minutos sigue activa y apunta a esta
candidata; el timer de informes periódicos permanece intacto.

La comprobación posterior confirmó el preflight vigente con Slurm COMPLETED,
salida cero y recibo exitoso. Verificó nuevamente todas las imágenes y la
cobertura por época de 125 páginas/5.212 líneas Berrutti, 23.774 líneas LUISA
y 1.991 composiciones. Su contrato coincide con la copia congelada local,
incluidos los cinco probes originales y los seis controles nuevos de atención.
El ensayo GPU cumplió `afterok` y está RUNNING, todavía sin recibo exitoso;
el último evento observado es preparación local del runtime. Eso no acredita
entrenamiento diagnóstico ni generación. No reenviar la cadena ni habilitar
baseline/train hasta aprobar el gate completo. La coordinación y la vigilancia
conservan esta evidencia vigente.

El combinado original continuó automáticamente desde sus 46 updates confirmados
y cerró los 48; ahora evalúa las 25 páginas de validación de la tercera época.
Todavía faltan ese cierre, selección y test final antes de verificar el resultado.
Se conservan todas sus fuentes y checkpoints; no se repite entrenamiento.
La comprobación actual registra 18 de 25 páginas de la validación final,
sin resumen CER cerrado todavía.


El ensayo de atención terminó FAILED antes de generar el primer token: el
wrapper recibía `mask`, pero la atención visual de Qwen llama la interfaz con
`attention_mask` por nombre. Slurm registró salida no cero y no hay recibo
exitoso. El modelo completo cargó en dos A100 y pasó el control de traslado
AdamW, pero no ejecutó ninguno de los tres updates diagnósticos ni acreditó
los controles de atención o las generaciones. No fue OOM ni una nueva prueba
de repetición. La corrección conserva la misma política de atención y cambia
sólo la compatibilidad del argumento de máscara. Una prueba de regresión
comprueba las llamadas por nombre de visión/decodificador, la delegación
intacta de máscara/escala y la restauración de la interfaz. Requiere cadena
nueva de preflight y ensayo completo; baseline/train siguen bloqueados.

El combinado cerró su tercera validación completa: 25 páginas, ninguna fallida
ni reintentada. Las tres épocas y 48 updates están confirmados; la selección
por validación completa eligió la época dos y el test está en curso. Aún requiere
cierre y control independiente final; no repetir entrenamiento ni evaluaciones
cerradas. La vigilancia permanece habilitada.


La corrección de interfaz tiene fuentes congeladas y cadena nueva enviada.
El preflight terminó con Slurm COMPLETED, salida cero y recibo exitoso. Verificó
125 páginas/5.212 líneas Berrutti, 23.774 líneas LUISA y las 1.991 composiciones,
con cobertura exacta de todas las líneas en cada época mixta. El contrato coincide
con la copia local congelada, incluidos los cinco probes y las tres actualizaciones
diagnósticas. El ensayo GPU cumplió `afterok`, obtuvo dos A100 y está RUNNING;
su último evento es preparación del runtime, sin recibo GPU exitoso todavía. Mantiene el gate
completo de memoria/proyección/atención. `make verify` pasó 213 tests, compilación,
lint y formato. Baseline y 64 updates reales siguen sin enviar. No reenviar el
ensayo que conserva su lugar en cola. La vigilancia apunta al nuevo puntero y
exige evidencia del intento vigente antes de actuar.

El combinado y su control CPU independiente terminaron con Slurm COMPLETED,
salida cero y recibo exitoso. Se verificaron las 48 actualizaciones contiguas
(1–9 del padre y 10–48 de la recuperación, sin repetir la décima), tres épocas
completas de 125 páginas/5.212 líneas, checkpoints/cursor con hashes y CRC,
inicialización/baseline y fuentes. Las tres validaciones completas seleccionaron
la época dos; el test descriptivo cerró las 25 páginas, ninguna fallida y cuatro
reintentadas, respetando el máximo de dos intentos. El control no entrenó ni
generó páginas nuevas. La coordinación conserva su prueba independiente.
El monitor inicialmente leyó un events.jsonl inexistente; ahora lee el completed.json
cerrado mediante un tipo explícito de recibo de auditoría. Distingue auditoría
exitosa de resultado/test completo y no expone rutas, texto OCR ni CER del control.
Páginas y combinado tienen resultados completos verificados; líneas conserva
su test incompleto y CER provisional, sin tercer intento. La vigilancia sigue
habilitada para la extensión y sus cierres pendientes; el informe periódico
permanece intacto.

El ensayo de atención con interfaz corregida terminó FAILED, salida no cero,
por generación incompleta. Completó los tres updates diagnósticos finitos,
restauraciones exactas y los seis controles de proyección y atención, sin OOM.
Cuatro probes finales llegaron a EOS; la misma composición de dos columnas
repitió ambos intentos antes y después de entrenar. Las entradas conservan
identidad, 48 líneas y referencias sin líneas duplicadas. La atención FP32
no resolvió el fallo; baseline y las 64 actualizaciones reales siguen sin enviar.

La candidata de convolución cambia sólo `luisa_mix.py` respecto de la copia congelada:
prueba convolución causal FP32 durante inferencia, con pesos y estado BF16,
actualización exacta de caché y salida BF16. La recurrencia nativa ya acumula
en FP32. No cambia entrenamiento, imágenes/píxeles/referencias, calendario,
probes, prompt, intentos ni recursos. La hipótesis conserva todos los controles
anteriores y añade seis controles de convolución con filas FP64 y estado exacto,
más cuatro casos por GPU. No acredita aún resolución. Las fuentes/resultados
anteriores y el corpus sellado se conservan intactos. `make verify` pasó 216
tests, compilación, lint y formato. El preflight nuevo terminó con Slurm COMPLETED,
salida cero y recibo exitoso: verificó todos los hashes, las 125 páginas/5.212
líneas Berrutti, las 23.774 líneas LUISA y las 1.991 composiciones con cobertura
completa en cada época. Su contrato coincide con la copia local congelada,
incluidos los cinco probes y los controles de convolución. El ensayo GPU cumplió
`afterok` y terminó FAILED con salida no cero, confirmado en Slurm y en sus
eventos actuales. Completó los tres updates diagnósticos finitos, restauración
exacta de adaptador/AdamW/calendario/RNG y los seis controles de cada política
de proyección, atención y convolución; no hubo OOM ni fallo de interfaz.
Cuatro probes finales llegaron a EOS. La misma página de dos columnas volvió
a repetir ambos intentos, antes y después del entrenamiento diagnóstico.
La convolución FP32 tampoco resolvió ese fallo. No existe recibo exitoso y el
gate permanece cerrado. El helper local
importa su propio validador congelado; el baseline nuevo y entrenamiento sólo
se envían después del gate completo. No reenviar esta cadena. La vigilancia
de cinco minutos sigue habilitada y el informe periódico permanece intacto.

La comprobación de envíos confirmó una sola cadena nueva y ausencia de baseline
o entrenamiento. Los comentarios de trabajos activos están en la cola viva,
pero la contabilidad puede devolver el campo Comment vacío. El control operativo
local de recuperación consulta ambos antes de ejecutar o reintentar el helper
congelado: sólo recupera recibos de envío ausentes de jobs propios identificados
de forma única y tras validar el gate vigente. No envía jobs ni modifica fuentes
congeladas. Sus instrucciones están en la vigilancia; una búsqueda vacía de
comentarios en sacct no autoriza reenviar.

El diagnóstico CPU independiente de entradas terminó con Slurm COMPLETED,
salida cero y recibo exitoso: verificó bytes y SHA256 de las 25.765 imágenes
mixtas únicas con cuatro lectores, sin modificar el corpus (7,06 GB, 249 segundos
en esa ejecución). Hubo efectos de caché en controles parciales; no extrapolar
un factor de velocidad. La fuente de desarrollo incorpora el verificador
completo y fases explícitas de verificación; las copias enviadas permanecen
intactas. Este control CPU no abre el gate GPU.

Tras el fallo de convolución se envió un diagnóstico del Qwen completo con
caché frente a recomputación de prefijos fijos, mediante teacher forcing de una
salida ya cerrada. No llama a `generate`, no añade intentos, CER ni updates.
Los recibos antiguos sólo guardan texto: la retokenización se identifica como
secuencia diagnóstica, sin afirmar que reconstruye los tokens originales.
Se conserva la imagen fallida, la inicialización común y las políticas vigentes.
Tiene fuentes y resultados nuevos, preflight completo nuevo y GPU `afterok`
de ese preflight, con las mismas dos A100. El corpus sellado ya figura COMPLETED
con salida cero en contabilidad; su trabajo antiguo salió del controlador,
por lo que no se impone una dependencia histórica inválida ni se reenvía la
preparación. El puntero privado `current-cache-prefix-diagnostic.json` conserva
la cadena y sus recibos, separado de la candidata vigente del experimento.
La comprobación local de la composición fallida encontró 48 identificadores
y referencias únicos, 24 recortes por columna, ninguna caja solapada y todas
dentro del lienzo. También verificó los hashes del PNG, referencias y RGB de
los recortes originales en el cache sellado, y comparó píxel por píxel las 48
regiones renderizadas con esos recortes escalados mediante LANCZOS: todas
coinciden exactamente. No escribió imágenes, no ejecutó OCR ni prueba capacidad
del modelo; el recibo distingue ese alcance del diagnóstico GPU.
Una diferencia numérica no demuestra causalidad ni sustituye el ensayo completo.
Consultar sus recibos y Slurm antes de considerar una corrección nueva; no
reenviar operaciones existentes ni repetir los tres entrenamientos originales.
El primer preflight del diagnóstico terminó Slurm COMPLETED, salida cero y
recibo exitoso, con todos los hashes y coberturas y los mismos cinco probes.
El control GPU posterior falló antes de cualquier forward: exigía salidas
visuales precalculadas, pero la clase Qwen instalada hereda
`input_modalities="text"` y su preparación nativa conserva los píxeles para
el forward. No fue OOM ni una nueva observación de repetición; cargó Qwen y
las dos A100, pero no comparó logits ni cachés.

Una copia nueva del control acepta esa ruta nativa sin inventar embeddings:
conserva píxeles y grilla idénticos en cada prefill completo y deja que la
interfaz nativa los descarte durante decode con caché. Mantiene la comparación
de dos prefills independientes en posición cero, la secuencia fija, posiciones,
máscaras, precisión y el máximo de nueve prefills. Un control local del
dispatcher nativo exacto comprobó las rutas con entradas ausentes/vacías y
precalculadas sin forward ni generación. Las fuentes de entrenamiento y
el corpus no cambian; `make verify` pasó 226 tests, compilación, lint y formato.
La nueva cadena cerró su preflight completo con Slurm COMPLETED, salida cero y
recibo exitoso: todos los hashes, coberturas y cinco probes coinciden. El
diagnóstico GPU terminó COMPLETED, salida cero y recibo exitoso, sin updates,
generaciones, intentos nuevos ni CER. Comparó ocho posiciones de la misma
secuencia retokenizada: los dos prefills en posición cero fueron idénticos,
y las posiciones, máscaras, rope deltas y píxeles coincidieron en todas. Los
top1 coinciden en las ocho; el máximo L2 relativo de logits fue 0,405. Esto no
demuestra causa de repetición ni equivalencia numérica.

La revisión independiente detectó una omisión del observador: las cachés
convolucionales y recurrentes nativas usan diccionarios, pero el control sólo
enumeraba tensores, listas y tuplas. Comparó 32 tensores de las 16 capas de
atención, dejando sin observar las 48 capas lineales. Su recibo acredita la
ejecución del diagnóstico, sin acreditar toda la caché ni abrir el gate.
Una raíz nueva corrige únicamente la observación: recorre mappings sin mutar
estados y exige el inventario derivado de las 64 capas y sus 128 tensores.
La regresión usa el constructor nativo exacto y rechaza seis formas de
cobertura incompleta. `make verify` pasó 227 tests, compilación, lint y formato.
Mantiene todas las fuentes de entrenamiento e inputs, nueve prefills máximos,
posiciones, políticas e inicialización, con dos A100 y cuatro horas. El nuevo
preflight y el diagnóstico terminaron Slurm COMPLETED, salida cero y recibos
exitosos. La revisión independiente comprobó las 64 capas y los 128 tensores
en ambas ramas de las ocho comparaciones, todos finitos y aislados. Posiciones,
máscaras, rope deltas y píxeles coinciden; los dos prefills en cero son idénticos.
Los top1 coinciden en las ocho posiciones, aunque los logits llegan a L2 relativo
0,405 y hay diferencias recurrentes desde la primera capa. Esto no demuestra
causa o resolución de repetición ni equivalencia numérica. No reenviar el control.
Consultar el puntero y recibos actuales; todas las fuentes y resultados
anteriores se conservan, incluida la evidencia completa de la omisión.
La vigilancia permanece habilitada. El diagnóstico sólo puede orientar una
corrección fundada y una candidata nueva con `make verify`, preflight y ensayo
representativo íntegros; no genera transcripciones, no hace updates ni CER y
no abre el gate de producción.

Un control separado `first-linear-kernel-replay-v1` aísla la primera recurrencia.
El primer control cerró preflight completo exitoso y Slurm GPU FAILED, salida
uno, tras completar ambos prefills del Qwen. Falló al comparar la calibración
nativa con la referencia, antes de los replays, sin guardar las magnitudes
de ese desacuerdo. Conserva los tensores capturados: Q/K/V del prefijo común
son idénticos; g y beta presentan diferencias pequeñas, sin demostrar causa
de repetición. Los cuerpos PyTorch de fallback pasan la calibración local,
pero el dispatcher instalado puede usar FLA; no son prueba de la misma
implementación efectiva en Ruche. No acreditar equivalencia ni resolución.
Una raíz nueva corrige únicamente la falta de evidencia: registra la
procedencia de los callables y métricas de ambos dispositivos antes de
rechazar por el umbral original de 1e-5, sin aflojarlo ni abrir el gate.
El puntero privado `current-linear-kernel-diagnostic.json` conserva la cadena
vigente de preflight completo y GPU `afterok`, mismas dos A100 y cuatro horas.
El control observa sin alterar argumentos ni resultados dos
prefills nativos del Qwen completo de la misma imagen, con cero y 622 tokens
diagnósticos fijos retokenizados. Contrasta las mismas entradas capturadas de
tres heads de la primera capa con los kernels nativos chunk/recurrent/split y
una recurrencia FP64 independiente, con calibración analítica en ambas GPU.
Compara además las entradas comunes de ambos prefills y exige cobertura de
64 capas/128 estados, finitud, identidad de parámetros, entradas y RNG.
No modifica fuentes de entrenamiento ni políticas, no genera, no entrena,
no crea intentos o CER ni acredita el gate. Una diferencia numérica en ese
control acotado tampoco demuestra causalidad. El control local comprobó
firma y llamada nativas, identidad de argumentos/resultados, límites y
restauración ante excepción; `make verify` pasó 227 tests y las comprobaciones
de compilación, lint y formato. La vigilancia incluye sus eventos y permanece
habilitada, con los informes periódicos intactos. La regresión adicional
comprueba la referencia FP64, preservación de métricas antes del rechazo y
siete rechazos de cobertura/umbral; la calibración local sólo usa los cuerpos
fallback exactos, nunca acredita la implementación efectiva ni Qwen completo.

La revisión fresca de la copia que registra métricas confirmó que el dispatcher
elige FLA. Su chunk falla la misma calibración en ambas A100: L2 relativo de
salida 9,76e-4 y de estado 4,88e-4; el recurrente pasa con 4,63e-8 y 2,32e-8.
Slurm cerró FAILED, salida uno, sin recibo exitoso ni replays completos.
Las fuentes FLA contienen productos Triton que permiten TF32; la observación
no demuestra todavía su contribución en las entradas reales ni la causa de
repetición. No aflojar el umbral ni sustituir el resultado por la calibración
local de los cuerpos PyTorch.

Una continuación privada nueva reutiliza los tensores sellados de esos dos
prefills, verifica sus SHA256 bajo un preflight completo nuevo y compara sólo
los mismos tres heads. No carga otro modelo ni hace forwards Qwen adicionales.
Mide FLA chunk/recurrent/split y los cuerpos fallback exactos, con copias FP32
identificadas explícitamente y referencia FP64 independiente; no modifica el
runtime de ningún modelo. Guarda todas las comparaciones antes de aplicar el
rechazo original de calibración. Si persiste el desacuerdo, conserva
`completed.json` con `success=false` y salida no cero aunque las mediciones
hayan terminado: no acreditar un control exitoso ni el gate. El control local
comprueba ese rechazo después de preservar los resultados. `make verify`
pasó 227 tests; conserva dos A100/cuatro horas, corpus y fuentes de entrenamiento
idénticos. Consultar el puntero y submission actuales y no reenviar la cadena.

La continuación acotada terminó FAILED/salida uno y conservó el rechazo nativo,
con todas las ocho variantes y comparaciones persistidas. El replay de estado
FLA reproduce exactamente el estado capturado. Su error relativo frente a FP64
es 4,02e-3; el cuerpo PyTorch nativo con copias FP32 obtiene 1,80e-7 en estado y
2,47e-7 en salida, con calibración analítica aprobada en ambas A100. Incluso las
copias FP32 de FLA conservan desacuerdos mayores: convertir entradas por sí solo
no resuelve este control. La evidencia justifica una candidata de inferencia,
pero no demuestra la causa ni la resolución de la repetición. No reenviar ese
control ni contabilizarlo como exitoso, entrenamiento o generación.

`linear_decode.py` aplica el cuerpo PyTorch chunk nativo fijado sólo durante
inferencia, con trabajo FP32, salida BF16 y estado recurrente FP32. Conserva
FLA recurrente, todas las fuentes de entrenamiento, datos, píxeles, referencias,
prompt, cinco probes y política de generación. Restaura interfaces y precisión
al salir, comprueba entradas/caché sin mutación y parámetros sin cambios. Exige
cuatro controles numéricos por A100 y seis controles durante generación; las
comparaciones FP64 se hacen antes del cast BF16 y mantienen el límite 1e-5.
Sólo un nuevo preflight completo y el ensayo Qwen íntegro pueden habilitar su
baseline y los 64 updates reales. Consultar el puntero vigente, Slurm y recibos;
la evidencia numérica acotada no abre el gate.

La candidata de prefill pasó 230 tests y ocho controles numéricos locales;
su preflight completo cerró Slurm COMPLETED, salida cero y recibo exitoso,
con todos los hashes y cobertura de 125 páginas/5.212 líneas Berrutti,
23.774 LUISA y 1.991 composiciones. El ensayo Qwen nuevo está RUNNING en Slurm,
con la dependencia satisfecha; baseline y entrenamiento siguen sin enviar.
El inicio no acredita carga completa, generación ni aprobación del gate.
Consultar siempre el puntero y Slurm frescos antes de actuar.

Se corrigió un fallo independiente del monitor local: los recibos cerrados
`performance_diagnostic` se estaban leyendo con un tail de 64 líneas, que
recortaba sus objetos JSON con formato y detenía la actualización del snapshot.
Ahora usa lectura acotada desde el principio y aísla los errores de un recibo,
sin acreditarlo ni perder los estados de los demás trabajos. Las regresiones
comprueban un recibo largo fallido y la conservación de otros controles ante
JSON parcial o inválido. `make verify` pasó 232 tests, compilación, lint y
formato. No cambia las fuentes congeladas ni la cadencia o destino de informes.

## Web de resultados de fine-tuning

`results_viewer.py` y `web/` construyen una interfaz estática independiente del
benchmark: train/validación/test de los tres modelos, baseline, épocas,
imagen/referencia/OCR y los 5.212 recortes train. La preparación del experimento
cuatro incorpora sus 23.774 líneas LUISA y 1.991 páginas sintéticas con
procedencia y referencias verificadas. No ejecuta inferencia ni repite
evaluaciones. Líneas conserva su test incompleto y CER provisional; la cuarta
prueba muestra datos preparados, sin inventar entrenamiento ni CER.

La release activa r4 responde al nuevo pedido de interfaz: imagen a la izquierda,
selector para una, dos o tres transcripciones, y debajo un solo panel de
diferencias con referencia y texto comparado independientes. Admite los tres
pares en ambas direcciones y mantiene cada carácter; una pareja completa no
hereda el estado provisional del tercer texto. Se recortaron introducciones y
textos genéricos. Train sigue sin OCR; la extensión sólo admite su baseline.

El bundle nuevo conserva 57.443 payloads byte a byte; sólo cambian los tres
assets y la fecha de construcción del catálogo, sin inferencia ni scoring.
`make verify` pasó 232 tests, compilación, lint y formato; seis pruebas Node
verifican alineación íntegra y asociación de baseline. El navegador pasó 73
controles sobre el bundle cerrado real y 76 de fixtures, sin errores, en
escritorio y móvil. Doctor: 15 checks correctos, sin warnings; dry-run correcto.

El usuario aprobó el destino, release ID y SHA-256 exactos de r4. La activación
transaccional terminó con salida cero, sin advertencias, y la verificación
independiente cerró el 12 de octubre a las 10:04 UTC: árbol remoto completo
idéntico, cinco archivos de entrada HTTPS con hashes correctos y 73 controles
de navegador en la URL pública, sin errores, en escritorio y móvil.
Los punteros `.private/current-results-viewer-candidate.json` y
`.private/current-results-viewer.json` registran `ACTIVE_VERIFIED` para r4;
conservan la aprobación, recibos y puntero anterior de r3. La identidad exacta
está en [RESULTS_VIEWER_RELEASE.md](RESULTS_VIEWER_RELEASE.md).
Seguir el CLI transaccional de la [guía](RESULTS_VIEWER.md); no editar bundles
cerrados ni escribir sobre los viewers históricos.

## Limpieza y otras ampliaciones

La limpieza autorizada canceló ambas corridas antiguas, retiró las recetas y
borró sus resultados privados. Se conservan datasets registrados, modelos base,
entornos y herramientas de datos/Archive. No restaurar launchers retirados.

Los informes periódicos y la vigilancia operativa están desactivados por pedido
del usuario. Las consultas manuales conservan separados los tres brazos, la
extensión y las evaluaciones previas. Dataset, crops, OCR, checkpoints, logs y
configuración privada siguen fuera de Git. No tocar trabajo ajeno de
datos/Archive ni entornos en uso.
