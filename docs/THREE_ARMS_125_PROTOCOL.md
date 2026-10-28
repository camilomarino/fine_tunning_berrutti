# Tres experimentos Berrutti, 125 páginas y tres épocas

Diseño confirmado el 9 de octubre de 2026. Es la única comparación vigente.
Los tres trabajos se envían detrás de un ensayo representativo y una evaluación
inicial compartida; el monitor indica si están en cola, activos o terminados.
`three_arm_plan.py` sella las entradas; `runner.py` ejecuta la receta común.

## Imágenes y conjuntos

Una imagen completa muestra una página Berrutti con su transcripción humana.
Una imagen de línea es un recorte de una sola línea de esa página con su propia
transcripción humana. LUISA es otro conjunto de imágenes de líneas; no entra
en ninguno de estos tres experimentos.

Se usan las 125 páginas train y todos sus 5.212 recortes, sin excluir páginas.
Validación: las mismas 25 páginas completas para los tres modelos.
Test: otras 25 páginas completas, también iguales para los tres.
Las páginas de validación/test y sus recortes nunca entran en entrenamiento.

Las cajas y etiquetas de las 125 páginas son válidas. En 34 páginas, las
referencias de página y la concatenación de las etiquetas de líneas presentan
discrepancias. Se registran sin modificar etiquetas ni descartar páginas.
Una diferencia entre modelos puede deberse también a esas etiquetas, por lo
que no se atribuye exclusivamente a la representación visual.

## Qué ocurre en cada época

Cada celda contiene una pasada por **todos** los ejemplos indicados.

| Experimento | Época 1 | Época 2 | Época 3 |
| --- | --- | --- | --- |
| Sólo páginas | 125 imágenes completas | Las mismas 125 completas | Las mismas 125 completas |
| Sólo líneas | Los 5.212 recortes de las 125 páginas | Los mismos 5.212 recortes | Los mismos 5.212 recortes |
| Combinado | 125 completas + sus 5.212 recortes | Las mismas 5.337 imágenes | Las mismas 5.337 imágenes |

En el combinado se presenta una página completa, después todas sus líneas y
luego la siguiente página. Ocurre así durante las tres épocas; no hay una fase
inicial de líneas seguida por otra sólo de páginas. El orden de las páginas
cambia entre épocas, usando la misma semilla en los tres modelos.

Los tres parten del mismo Qwen fijado y de idéntica inicialización LoRA nueva.
No heredan checkpoints entrenados. No hay preparación adicional ni cuarta época.
La evaluación del modelo inicial ocurre antes de la primera época y no cuenta
como entrenamiento. Se registran configuración, semillas, hashes y eventos.

## Steps y pesos

Un step es una actualización de los pesos, no una imagen ni una época.
Se acumula el trabajo de ocho páginas de origen antes de actualizar. Según
el experimento, cada página de origen aporta su imagen completa, todas sus
líneas o ambas representaciones.

- Sólo páginas: imagen completa, peso 100 %.
- Sólo líneas: todas las líneas de esa página juntas, peso 100 %.
- Combinado: imagen completa 50 % y todas sus líneas juntas 50 %.

Las líneas de una página se ponderan por tokens supervisados. Cada época
comprende 15 grupos de ocho páginas de origen y un grupo final de cinco:
16 actualizaciones por época y 48 por modelo. Igualar actualizaciones no
iguala cantidad de imágenes, tokens ni tiempo GPU.

## Evaluación y ejecución

La línea base inicial se evalúa **una sola vez** en las 25 páginas completas
de validación y las 25 de test, y se comparte entre los tres modelos. Sólo se
reutiliza si coinciden pesos iniciales, adaptador, imágenes, referencias,
processor, prompt, política de generación y puntuación, registrados por hash.
No se reutilizan predicciones entre checkpoints distintos.

Después de cada época se evalúan las 25 páginas completas de **validación**.
Una vez evaluadas las tres épocas, se selecciona el checkpoint con menor CER
**de validación completa** entre las épocas 1, 2 y 3. Sólo ese checkpoint pasa
por las 25 páginas de **test**, una vez por modelo; el test es descriptivo.
Un empate se resuelve por la época más temprana. Una validación incompleta
no puede ganar; si ninguna es completa, no hay checkpoint seleccionado.
Se mantiene la misma generación y puntuación, con máximo dos intentos por
página, registro de fallos/reintentos y CER provisional cuando falten páginas.

| Trabajo de evaluación para los tres modelos | Páginas procesadas |
| --- | ---: |
| Una línea base compartida: 25 de validación + 25 de test | 50 |
| Validación: 3 modelos × 3 épocas × 25 páginas | 225 |
| Test del checkpoint elegido: 3 modelos × 25 páginas | 75 |
| **Total, sin contar reintentos** | **350** |

Evaluar también test tras cada época produciría 500 evaluaciones de página,
incluso compartiendo la línea base. Este calendario evita 150 (30 %), sin
reducir entrenamiento ni validación. No significa 30 % menos de tiempo total:
la duración por página, los reintentos y el entrenamiento se miden aparte.
`evaluation_schedule()` y `select_test_epoch()` implementan el calendario
compartido y la selección; el runner ejecuta ese calendario.

El runner implementa optimizador y calendario comunes, proyección FP32,
checkpoints por step y continuación bajo Slurm que restaura adaptador,
optimizer, scheduler y RNG. Se entrena sólo en Ruche, con dos A100 por trabajador.
La dependencia Slurm exige que termine correctamente el ensayo pequeño, que
incluye una página con discrepancias entre etiquetas y otra densa. El ensayo
comprueba las tres estrategias, checkpoint y generación completa de una página
de cada split por estrategia; sus CER son diagnósticos. Después se evalúa el
modelo inicial sin actualizaciones, antes de habilitar los tres brazos.
Que gane el combinado es una hipótesis que se medirá, no una garantía.

## Perfil común y tiempo

Los tres usan BF16, atención SDPA y proyección visual FP32. No se reducen
resolución, transcripciones, épocas ni conjuntos. La caché visual congelada
no superó el control de identidad y queda excluida. Tampoco se instala ningún
kernel nuevo en entornos existentes. Los diagnósticos de optimización se siguen
por separado y no aportan resultados científicos de estos tres modelos.

La caché CPU de 5.212 recortes conserva exactamente píxeles RGB y etiquetas
humanas; se construyó en 14,6 s con ocho CPU y ocupa 140 MB. El runner la consume
sin reconstruir los crops y conserva en RAM CPU las entradas procesadas para
reutilizarlas en las épocas siguientes. No almacena estados del decoder.

El paquete de las mismas librerías se copia al SSD del job antes de cargarlo.
El diagnóstico CPU completó la importación en 51 s, frente a 505 s desde el
entorno compartido; el job con copia duró 2 min 18 s. La ventaja observada es
de arranque; no acredita un multiplicador del entrenamiento o la generación.
El ensayo GPU obligatorio valida las mismas librerías en ese perfil.

El cupo efectivo observado el 9 de octubre permite cuatro A100 simultáneas:
dos trabajadores independientes de dos GPU. Se envían los tres brazos tras el
baseline; la cola decide cuáles comienzan primero. El runner mantiene el modelo
cargado entre entrenamiento y validación dentro de cada segmento. No reparte
una actualización de pesos entre trabajadores de recetas distintas.

Cada segmento solicita ocho CPU, 96 GB, 30 GB de SSD y 24 horas. El runner cede
tras 21 horas en un límite de grupo o página y envía su continuación, hasta ocho
segmentos. El checkpoint contiene pesos LoRA, AdamW, calendario y RNG; un recibo
con hash verifica sus bytes. Los intentos de generación ya reservados cuentan
contra el máximo de dos aunque se interrumpa el job. No se repiten páginas
cerradas ni actualizaciones confirmadas al continuar.

`performance.py` también implementa reparto 13/12 y agregación CER micro con
cobertura e identidad estrictas. Esa opción no está activada en este lanzamiento;
el cupo se dedica al paralelismo entre brazos. No hay una estimación fiable de
la duración total hasta medir el ensayo y las primeras épocas completas.

## Ampliación futura con LUISA

Una vez elegido el mecanismo que mejor funcione con Berrutti, el cuarto
experimento incorporará más datos mediante dos variantes:

1. Añadir imágenes individuales de líneas LUISA al mecanismo elegido.
2. Construir páginas sintéticas con esas imágenes de líneas e incorporarlas
   como páginas completas.

Una página sintética coloca recortes reales sobre un lienzo y concatena sus
etiquetas humanas en el orden de colocación. No genera letras ni usa OCR como
etiqueta. Debe conservar procedencia, posiciones, escala y referencias privadas.

Antes de usar LUISA se auditarán solapamientos contra las 175 páginas Berrutti
conocidas y se reservarán sus 2.163 líneas del test publicado. Un filtro de líneas
no certifica independencia documental. El tamaño, la proporción y las épocas
de esta ampliación quedan por definir. Por ahora no tiene código de entrenamiento,
trabajos enviados ni resultados; los tres experimentos principales siguen sin LUISA.
