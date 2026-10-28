# Experimento cuatro: Berrutti, LUISA y composición de páginas

Extensión autorizada el 10 de octubre de 2026. Se ejecuta por separado y no
modifica las fuentes, resultados o checkpoints de los tres brazos Berrutti.
Pregunta exploratoria: cómo evoluciona el reconocimiento de páginas al combinar
páginas reales, líneas de ambos conjuntos y composiciones con orden conocido.

## Datos y alcance de la evaluación

Se conservan las 125 páginas train Berrutti, sus transcripciones humanas completas
y sus 5.212 crops. La opción `include-all` incorpora las 23.774 líneas LUISA:
21.611 del train publicado y 2.163 del test publicado. Usa el campo humano del
CSV original, nunca la salida Tesseract. Esta inclusión literal sigue la nueva
petición del usuario. Una alternativa explícita `exclude-known-overlap` reserva
el test LUISA y excluye las coincidencias detectadas; no se activa silenciosamente.

Hay solapamientos conocidos con el test Berrutti. La preparación audita todas
las líneas contra las 50 páginas de validación/test: texto normalizado exacto,
subcadenas largas y hashes de crops normalizados visualmente. Guarda cantidades
y motivos, sin publicar datos. LUISA carece de identificadores de documento de
origen: este control no certifica independencia documental. Con `include-all`,
los CER son exploratorios y el test está contaminado; no acreditan una mejora
independiente frente a los tres modelos originales. Se preservan los 25 ejemplos
de cada split y sus referencias. Nunca se selecciona mediante test.

## Síntesis reproducible

Cada época mixta compone páginas nuevas usando todas las líneas seleccionadas
LUISA y todas las líneas train Berrutti, exactamente una vez en composiciones
por época. Además, todas esas líneas se presentan individualmente. Cada lienzo
tiene dimensiones idénticas a una página train Berrutti real. Alterna una y dos
columnas con márgenes, alturas, anchuras, separaciones, sangrías y espacios de
párrafo variables; puede agregar reglas no textuales y un fondo tenue.

Cada crop se coloca completo, sin cortar ni borrar caracteres, conservando su
proporción al redimensionarlo para el layout. El orden es de arriba abajo en la
columna izquierda y luego en la derecha. El target concatena las transcripciones
humanas en ese mismo orden. No se inventa texto. Un manifest privado registra
semillas, coordenadas, orden, procedencia y hashes de imágenes y referencias.
Una comprobación de cobertura impide perder o duplicar líneas en la síntesis.
Los originales no se modifican. El límite de 48 líneas por composición reparte
el excedente entre más páginas; no descarta líneas.

La preparación reutiliza los crops Berrutti verificados y lee cada original
LUISA una vez para una caché de bytes sin pérdida, con memoria acotada. Planifica
con el mismo RNG y compone páginas independientes en cuatro CPU. Comprueba contra
los originales la identidad exacta de PNG, targets y procedencia en una página
de cada layout; no usa una caché de embeddings ni cambia los píxeles. Publica
avances y un recibo cerrado por página, además del manifest y las coberturas finales.
La preparación dispone de cuatro horas en la partición CPU adecuada: el primer
intento agotó su hora durante la síntesis, después de auditar todas las líneas.
Su ensayo GPU dependiente se canceló sin ejecutarse; ese corte no acredita un
fallo CUDA ni un ensayo representativo realizado.

## Entrenamiento y seguimiento

Inicialización nueva con los mismos pesos base y semilla de la comparación.
Tres épocas mixtas completas, luego una cuarta época de consolidación usando
sólo las 125 páginas y 5.212 líneas Berrutti. Cada época tiene 16 actualizaciones,
acumulando ocho unidades de origen (cinco en el último grupo): 64 en total.
Las líneas LUISA y páginas sintéticas se distribuyen de manera equilibrada entre
las 125 unidades. Cada unidad intercala sus cuatro familias de ejemplos.

En las épocas mixtas cada familia pesa 0,25: página real, líneas Berrutti,
líneas LUISA y páginas sintéticas. Dentro de una familia se pondera por tokens
supervisados. La consolidación pondera página/líneas Berrutti por mitades.
Usar más líneas cambia datos, cómputo y objetivo: no es una cuarta representación
comparable con igualdad de datos a los tres brazos originales.

Se ejecuta un ensayo nuevo del Qwen completo en dos A100: entrenamiento finito
con las cuatro familias, recarga exacta de adaptador/AdamW/calendario/RNG y
generación completa en páginas reales, composiciones de una/dos columnas y una
línea LUISA. Requiere Slurm COMPLETED y recibo exitoso. Después se mide un baseline
propio con cero updates antes de habilitar entrenamiento. No se reutiliza un
baseline de identidad de runtime diferente.

La candidata de muestreo oficial autorizada pasó ese ensayo el 13 de octubre:
preflight y GPU Slurm COMPLETED, salida cero, recibos completos y verificador
LOCAL congelado exitoso sobre evidencia fresca. Conservó la inicialización
común, los tres updates diagnósticos con restauración exacta, todos los controles
numéricos y los cinco probes finales hasta EOS, sin fallos ni reintentos. Las
36 fuentes congeladas quedaron verificadas también en remoto. La línea base
propia cerró Slurm COMPLETED, salida cero y recibo exitoso, con 25/25 páginas
en validación y test: CER 2,904 % y 5,538 %, cero fallidas y cero/tres reintentadas,
respectivamente. La comprobación independiente recalculó las 50 páginas y
verificó 53 intentos totales, máximo dos por página, semillas y contratos intactos,
sin entrenamiento.

El entrenamiento cerró el 19 de octubre de 2026. La auditoría independiente
del 22 de octubre confirmó sus diez segmentos Slurm COMPLETED/salida cero,
64 actualizaciones contiguas finitas y 16 por época, sin pasos repetidos.
Se cargaron y verificaron los 64 checkpoints, sus hashes, adaptadores y momentos
AdamW, calendario y estados/cursor RNG. Las tres épocas mixtas recorren todas
las 125 páginas, 5.212 líneas Berrutti y 23.774 LUISA; se conservan las 1.991
composiciones y la consolidación Berrutti completa. Se verificaron de nuevo los
bytes y hashes de las 25.765 imágenes mixtas únicas, sin modificar entradas.

Las cuatro validaciones finales completaron 25/25 páginas sin fallidas ni
reintentadas: CER 2,589 %, 2,248 %, 2,173 % y 2,048 %. La selección mínima
corresponde a la época 4 y excluye los doce controles intermedios. El test se
ejecutó una sola vez para ese checkpoint: 25/25 completas, CER 3,220 %, cero
fallidas y tres reintentadas, con máximo dos intentos por página.

La [auditoría comparativa final](FINETUNING_AUDIT_20261022.md) conserva métricas,
incertidumbre y coste. El modelo mejora su baseline propia, pero no demuestra
una ventaja independiente de añadir LUISA: el test tiene contaminación conocida
y también cambiaron épocas, objetivo, precisión y política de generación.
Los probes técnicos no sustituyen estas evaluaciones de calidad.
No se requieren más continuaciones, updates ni intentos para cerrar esta corrida.
El seguimiento automático sigue retirado desde el 14 de octubre; las consultas
posteriores son manuales, a petición del usuario.

Los siguientes párrafos conservan antecedentes cerrados; los estados actuales
se consultan en el puntero privado vigente y sus recibos.

El primer ensayo ejecutado tuvo pérdidas y gradientes finitos y restauró su
checkpoint, pero la composición más extensa de dos columnas agotó los dos
intentos por repetición. Terminó FAILED y no habilitó baseline ni entrenamiento.
La siguiente candidata conserva exactamente las mismas composiciones y probes,
incluida esa página; no reduce datos o píxeles ni cambia la política de generación.
Añade un control de esa página con la inicialización original para distinguir
una dificultad previa de una regresión tras entrenar. Su ensayo realiza dos
updates mixtos y uno de consolidación, con restauración tras cada uno, y contrasta
todos los momentos de AdamW, calendario y RNG. Son tres updates diagnósticos,
sin relación con las 64 actualizaciones del modelo nuevo. Las cinco generaciones
finales siguen siendo obligatorias; se guardan todos los fallos y no se aprueba
la candidata mientras alguna esté incompleta. La preparación sellada se reutiliza
sin modificarla, tras verificar nuevamente sus hashes bajo Slurm. Esta candidata
prueba preparación adicional; todavía no demuestra resolver la repetición.

Ese ensayo confirmó repetición también con los pesos iniciales, antes de cualquier
update. Su primera actualización mixta y la restauración exacta terminaron, pero
el segundo backward agotó memoria GPU: los momentos AdamW ya inicializados
ocupaban espacio durante el cálculo de gradientes. No llegó a la consolidación
ni a las cinco generaciones finales, y no habilitó entrenamiento.

La candidata siguiente conserva esos tres updates y los cinco probes. Traslada
sólo los momentos AdamW a RAM entre actualizaciones y los devuelve, sin cambiar
dtype ni valores, al dispositivo de cada parámetro antes de `AdamW.step`.
Los pesos del Qwen completo y el backward permanecen en las dos GPU. Conserva
contadores, calendario, RNG, pérdida y todos los datos/píxeles/referencias; no
modifica generación ni recursos. Un control GPU compara cuatro pasos y todos
los momentos contra AdamW residente, con igualdad exacta. Cada update Qwen
debe restaurar exactamente su checkpoint y confirmar los momentos en RAM.
Los eventos guardan memoria antes/después del traslado. Este cambio requiere
el nuevo ensayo completo; liberar memoria no acredita resolver la repetición.

Ese ensayo completó los tres updates finitos y restauró exactamente adaptador,
todos los momentos AdamW, calendario y RNG, sin OOM. Las dos páginas reales,
la composición de una columna y la línea LUISA llegaron a EOS; la misma
composición de dos columnas repitió en los dos intentos. Slurm terminó FAILED
y el gate permanece cerrado. La repetición previa al entrenamiento sigue
impidiendo atribuir el fallo a los updates.

La candidata de precisión calcula la proyección de salida en FP32 únicamente
durante generación. Conserva los pesos BF16 almacenados, sus valores exactos,
el entrenamiento, los cinco probes y todos los parámetros y criterios de
generación. Es una hipótesis sobre la precisión de los scores del decodificador,
no una causa demostrada ni una resolución acreditada. Registra empates y cambios
de argmax frente a la salida BF16 nativa en intervalos fijos, sin consultar texto
o CER ni generar intentos adicionales. Comprueba la copia exacta de pesos y la
referencia lineal FP32; restaura el forward original incluso si falla una página.
Requiere un nuevo preflight y el ensayo Qwen completo bajo Slurm antes de enviar
baseline y entrenamiento propios. Los tres modelos Berrutti conservan su runtime.

El ensayo de proyección FP32 también terminó FAILED por repetición en ambos
intentos de la misma página de dos columnas, antes y después de entrenar.
Pasaron los seis controles de precisión, los tres updates diagnósticos y sus
restauraciones exactas, sin OOM; cuatro probes finales completaron generación.
La proyección de salida por sí sola no resolvió el fallo y no habilitó baseline.
La siguiente candidata mantiene ese control y prueba SDPA matemático FP32 sólo
en la atención completa del decodificador durante inferencia. Conserva las
máscaras, escala, pesos y cachés almacenados, atención visual, entrenamiento,
datos y criterios de generación. Contrasta filas acotadas con referencia FP64,
exige salidas finitas, verifica que los parámetros no cambien y restaura la
interfaz original incluso ante un fallo. Es una hipótesis adicional, todavía
sin aprobación GPU; mantiene los cinco probes obligatorios y los tres updates.


El primer ensayo de esta candidata de atención falló por una incompatibilidad
de interfaz antes de generar tokens: Qwen pasa `attention_mask` por nombre y
el wrapper requería `mask`. No hubo OOM, updates diagnósticos ni prueba de la
hipótesis de precisión. La copia siguiente acepta el nombre nativo y conserva
la máscara, escala y todas las operaciones/controles anteriores. Necesita
preflight y ensayo GPU nuevos completos; el gate sigue cerrado.

El ensayo con interfaz corregida ejecutó las tres actualizaciones finitas y
restauraciones exactas, sin OOM. Pasaron los seis controles de atención FP32
y de proyección; cuatro probes finales llegaron a EOS. La misma composición
de dos columnas repitió ambos intentos, también con los pesos iniciales, por
lo que la atención FP32 no resolvió la dificultad y el gate sigue cerrado.
La página, sus 48 líneas y referencias conservan identidad; no hay líneas
de referencia duplicadas. El acumulador de la recurrencia ya usa FP32 en el
runtime fijado, mientras su convolución causal usa BF16.

La nueva hipótesis calcula sólo esa convolución causal durante inferencia en
FP32, con productos y sumas explícitos y almacenamiento temporal acotado por
canales. Conserva pesos y caché BF16, incluida la actualización exacta del
estado convolucional, y devuelve BF16 a la siguiente operación. Mantiene
entrenamiento, datos, píxeles, referencias, cinco probes, prompt y política de
generación. Restaura las dos funciones originales incluso ante un fallo y
comprueba identidad, almacenamiento, versión, dtype y dispositivo de parámetros.
El gate añade seis controles con referencia FP64 acotada, salidas finitas,
prefill/decode y estado exacto, además de un control pequeño de cuatro casos por
GPU. No sustituye los controles anteriores ni el ensayo del Qwen completo.
La hipótesis requiere una cadena nueva en dos A100; aún no demuestra resolver
la repetición ni habilita baseline o las 64 actualizaciones reales.

Una comprobación CPU independiente terminó con Slurm COMPLETED, salida cero y
recibo exitoso: leyó íntegramente las 25.765 imágenes mixtas únicas (7,06 GB) y
contrastó todos sus SHA256 usando cuatro lectores concurrentes. No omitió
archivos ni alteró el corpus. La lectura completa concurrente duró 249 segundos;
los controles parciales detectaron efecto de caché, por lo que no se extrapola
un factor de aceleración a otros jobs. La fuente de desarrollo incorpora ese
verificador completo y fases explícitas para distinguir hashes de staging y
carga GPU. Las copias congeladas ya enviadas conservan su verificación original;
esta comprobación CPU no acredita generación ni abre el gate del experimento.

Se evalúan las 25 páginas completas de validación después de las actualizaciones
4, 8, 12 y 16 de cada época. Las tres primeras son controles de avance; sólo las
cuatro validaciones al cerrar las épocas participan en la selección. El test de
25 páginas se ejecuta una vez para el checkpoint seleccionado, al terminar las
cuatro épocas. El usuario autorizó probar en esta extensión la política instruct
oficial de la revisión fijada: muestreo, temperatura 0,7, top-p 0,8, top-k 20,
min-p 0,0, penalización de presencia 1,5 y penalización multiplicativa 1,0.
La presencia se cuenta una vez por token generado distinto, excluyendo el prompt.
Las semillas dependen sólo de la semilla común, página, hash de imagen e intento;
se restaura el RNG global incluso ante una excepción o al reanudar el segundo
intento. Baseline y checkpoints usan esta misma política. El nuevo recibo exige
seis controles de muestreo y la restauración exacta de interfaces/RNG, además de
todos los controles anteriores y los mismos cinco probes completos. Sólo un
preflight y ensayo nuevos íntegros habilitan entrenamiento. Se conservan píxeles,
EOS, guard de repetición, presupuestos y máximo dos intentos. CER con páginas
incompletas es provisional. Los tres modelos originales conservan su política.

Cada actualización queda confirmada por checkpoint cerrado. Las continuaciones
restauran AdamW, calendario de 64 pasos y RNG, y no repiten updates confirmados.
Los segmentos tienen un presupuesto blando de 16 horas dentro de jobs de 24
horas; sólo ceden entre grupos o páginas de evaluación. Se registra un máximo
de 16 segmentos. No se cambian recursos o datos para esconder fallos.

`luisa_prepare.py` sella datos y composición en CPU bajo Slurm. `luisa_mix.py`
ejecuta preflight, ensayo, baseline y entrenamiento; `run_luisa_mix.sbatch`
conserva el allocator ya verificado en Ruche y continúa desde el cursor.
Todo payload, manifest, checkpoint y evidencia permanece fuera de Git.
La consulta manual incorpora esta extensión en un campo separado. Los timers
de monitor, informes al chat y vigilancia se retiraron el 14 de octubre por
pedido del usuario; no reactivarlos sin una nueva petición explícita.
