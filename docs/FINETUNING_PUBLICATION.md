# OCR Diff Viewer · fine-tuning Qwen–Berrutti

El usuario descartó la portada publicada en r1. La sustitución publicada usa
**la copia exacta del OCR Diff Viewer v2** indicado por el usuario:
<https://iie.fing.edu.uy/~cmarino/ocr/ocr_diff_viewer_v2/>.
La web nueva abre directamente en su matriz de resultados y su explorador,
sin portada, iframe ni visores incrustados.

Está publicada y verificada en
<https://iie.fing.edu.uy/~cmarino/ocr/qwen_berrutti_comparison/>.
La release r4 muestra un único **Qwen sin entrenamiento**, con los resultados
deterministas originales. El usuario pidió retirar la segunda evaluación sin
ajuste y las etiquetas «referencia Berrutti/LUISA». Se conservan los cuatro
ajustes y sus IDs y enlaces. La evaluación retirada permanece en la auditoría
y en las releases anteriores, sin sustituirla por la determinista en el análisis.
El v2 de referencia permanece congelado
y de sólo lectura; su manifest público conserva el hash original.

## Interfaz de referencia

El manifiesto público de v2 y su copia local coinciden exactamente:
`9994fe5dc725299d9f2f6e0f171d5569820412c94ec3267907facaa24cb65d6e`.
Se reutilizan sin cambios `app.js`, `styles.css`, `extensions.js`,
`extensions.css`, `loader.js` y `favicon.svg` de esa publicación, no los assets
actuales del checkout. El HTML conserva toda la estructura original; sólo
cambian el hash del catálogo y las etiquetas que cargan el contexto del
fine-tuning. Retirar esas adiciones recupera exactamente el HTML publicado.

La matriz, búsqueda, ordenamiento, CSV, imagen, zoom, redimensionado, referencia,
OCR, diferencias CER/WER y Raw/Norm son los de v2. Una extensión pequeña añade
el cambio validación/test, un botón para alternar base/ajuste conservando la
página y un diálogo con el CER micro oficial. Mantiene los avisos de generación
incompleta y LUISA exploratorio. Las figuras de entidades, sin evidencia para
esta comparación, no se presentan como controles activos. El ajuste móvil
mantiene los controles dentro de la pantalla.

## Resultados preservados

La [auditoría final](FINETUNING_AUDIT_20261022.md) conserva las mismas 25 páginas
de validación y 25 de test. El visor conserva sus referencias, píxeles y 250
predicciones seleccionadas de los cinco modelos visibles:

| Nombre visible | Datos de entrenamiento | Checkpoint mostrado |
| --- | --- | --- |
| Qwen sin entrenamiento · 0 épocas | Sin entrenamiento adicional; generación determinista sobre Berrutti | Base compartida |
| Berrutti: sólo páginas · 3 épocas | 125 páginas completas | Época 2 |
| Berrutti: sólo líneas · 3 épocas | 5.212 recortes de líneas | Época 2; test provisional 24/25 |
| Berrutti: páginas + líneas · 3 épocas | 125 páginas y sus 5.212 líneas | Época 2 |
| Berrutti + LUISA + sintéticas · 3+1 épocas | Berrutti completo, 23.774 líneas LUISA y sintéticas; tres épocas mixtas y una sólo Berrutti | Época 4 |

Los nombres se muestran en la matriz, selector, título OCR, fichas y resultados.
La búsqueda y el orden alfabético usan esos nombres. Los IDs internos se
conservan para no cambiar los enlaces a páginas y modelos. Los detalles indican
las cantidades anteriores; «Época elegida» distingue el checkpoint de la duración
del entrenamiento indicada en el nombre.

La prueba con LUISA mantiene su configuración de generación distinta: no se
reasigna a la base determinista ni se ofrece un botón que afirme esa asociación.
La matriz original muestra medias/medianas por página. El botón **Resultados**
muestra el CER micro auditado y las páginas fallidas/reintentadas. El test de
líneas conserva una página incompleta y sus dos intentos agotados. LUISA sigue
separado como exploratorio, con test contaminado y receta distinta.
No se repitieron entrenamiento, OCR, intentos ni evaluaciones. Las etiquetas de
los modos cambian para que el v2 original las muestre de manera legible; sus
textos y puntuaciones permanecen iguales.

## Release activa y verificada

| Campo | Valor |
| --- | --- |
| Release | `qwen-berrutti-ocr-diff-20261026-r4` |
| Manifest SHA-256 | `e0e5b768ee19ea77d50413271870b64c742d51d0d20ab45ddf72d3bbca94359c` |
| Archivos, incluido el manifest | 387 |
| Bytes | 46.461.985 |
| Policy | `ascii-text-v1` |
| Páginas / modelos / predicciones | 50 / 5 / 250 |
| Configuración | `configs/publish/ampere-finetuning-comparison.toml` |

Doctor y dry-run canónicos pasaron sin advertencias. `make verify` pasó 272
tests, con tres omisiones existentes, compilación, lint y formato. El navegador
pasó 38 comprobaciones locales y las mismas 38 sobre la URL pública, sin errores,
incluidas las 250 predicciones y la vista móvil. La verificación independiente
contrasta las 250 predicciones, 50 referencias e imágenes con la publicación
auditada anterior, conservando los scores, estados e intentos exactos. Se retiran
de la publicación los 52 fragmentos de la evaluación sin ajuste excluida.
Los seis assets originales de v2 siguen byte a byte idénticos. Los bundles, datos y
recibos permanecen fuera de Git; los bytes anteriores se conservan como evidencia.
La activación canónica terminó con salida cero, sin advertencias, usando
exactamente el bundle, destino, release y SHA-256 fijados para el cambio
solicitado. La comparación
completa del árbol remoto no encontró diferencias; el manifest público y ocho
archivos consultados por HTTPS coinciden en tamaño y hash. Los punteros privados
registran `ACTIVE_VERIFIED` y conservan la instrucción del usuario y los recibos.
El bundle quedó congelado antes de la activación. No se reactivaron automatizaciones.
