# Release publicada en Ampere

## Comparación final independiente · 26 de octubre de 2026

El usuario descartó la portada y los visores incrustados de r1. La
[sustitución publicada y verificada](FINETUNING_PUBLICATION.md) utiliza directamente la copia
exacta del OCR Diff Viewer v2. La release r4 muestra las mismas 50 páginas y
250 predicciones auditadas: un único Qwen sin entrenamiento, con generación
determinista, y los cuatro ajustes. Está activa en
<https://iie.fing.edu.uy/~cmarino/ocr/qwen_berrutti_comparison/>.
El árbol remoto y el manifest público coinciden con el bundle aprobado; las
38 comprobaciones del navegador público pasaron sin errores. Se conservan los
nombres con datos y épocas y los IDs de las cinco entradas. La segunda evaluación
sin ajuste fue retirada sólo de esta publicación a pedido del usuario. R1 permanece
como evidencia de la publicación anterior. No cambia el destino ni el snapshot
de r5 descritos a continuación.

## Release activa r5 · 16 de octubre de 2026

La release `qwen-berrutti-finetuning-20261016-r5` fue aprobada explícitamente,
activada y verificada en Ampere. El árbol remoto completo coincide con el
bundle aprobado y los cinco archivos de entrada consultados por HTTPS
conservan sus tamaños y hashes exactos.
Su manifest SHA-256 es
`7f5fd32437f2c27aac0c3b3f7f3af6990e8f65a3a97d88803559d4a0f18a9ca2`.

- Las cuatro tarjetas resumen los datos y épocas de cada receta.
- La extensión incorpora su baseline propia y seis validaciones cerradas,
  incluida la primera de fin de época. Su estado observado es época 2 de 4,
  25/64 actualizaciones y 17.198/29.773 imágenes confirmadas en la segunda.
- Los controles intermedios se distinguen de los cierres seleccionables;
  todavía no hay selección ni test final de la extensión.
- Conserva íntegros los resultados de los tres modelos originales y todos
  los payloads de datos anteriores. El test incompleto de líneas sigue provisional.
- 22 evaluaciones cerradas, 550 documentos evaluados, 57.648 archivos.
- `make verify`: 244 tests; compilación, lint y formato correctos. Navegador:
  78 comprobaciones con datos reales locales, 78 sobre la URL pública y 76
  con fixtures, sin errores. Doctor:
  15 comprobaciones correctas, cero advertencias; dry-run correcto.
- Publicación estática consultada manualmente. No se reactivan llamadas
  automáticas, vigilancia ni informes periódicos.

Los punteros activo y de candidata registran `ACTIVE_VERIFIED`. La operación
conserva la aprobación exacta, el puntero anterior de r4 y todos los recibos.
URL: `https://iie.fing.edu.uy/~cmarino/ocr/qwen_berrutti_finetuning/`.

## Release anterior r4

Estado: activación aprobada por el usuario, completada y verificada el
12 de octubre de 2026 a las 10:04 UTC.

- Destino público: `https://iie.fing.edu.uy/~cmarino/ocr/qwen_berrutti_finetuning/`.
- Configuración: `configs/publish/ampere-finetuning.toml`.
- Release: `qwen-berrutti-finetuning-20261012-r4`.
- SHA-256 del manifest: `8391b6c7614846b0cd034dcbf18bc1376f26bbff3a1f0c28c84c98bb2f014108`.
- Árbol publicado: 57.448 archivos, 7.406.418.382 bytes, idéntico al bundle aprobado.
- Cobertura: 175 páginas Berrutti, 5.212 recortes train, 14 evaluaciones cerradas
  con 350 recibos de página, 23.774 líneas LUISA y 1.991 páginas sintéticas.
- Los tres modelos originales conservan tres épocas y 48 actualizaciones cada
  uno. Páginas y combinado tienen test completo; líneas conserva una página
  incompleta y su CER provisional, sin generación adicional.
- La cuarta prueba muestra datos preparados, sin entrenamiento ni CER; sus
  resultados futuros serán exploratorios, con contaminación conocida del test.

El CLI transaccional terminó con salida cero. La comparación completa del árbol
remoto no encontró diferencias; el manifest y los cuatro archivos de entrada
consultados por HTTPS coinciden en tamaño y SHA-256. El navegador sobre la URL
pública pasó 73 controles, sin errores, en escritorio y móvil: navegación de
modelos, imágenes/transcripciones, recortes, sintéticas, estado provisional y
las siete vistas de transcripciones y seis parejas dirigidas en un panel único.

La operación privada conserva el bundle y los recibos de construcción, pruebas,
doctor, dry-run, activación y comprobaciones posteriores. El puntero local de
resultados registra `ACTIVE_VERIFIED` y permite recuperar exactamente esos bytes.
No reconstruir ni editar este bundle después de aprobar su hash.
La [guía](RESULTS_VIEWER.md) describe el procedimiento transaccional de
activación y las comprobaciones posteriores. Esta release usa un destino
independiente del benchmark y no altera sus viewers históricos.

## Release anterior r3

La release `qwen-berrutti-finetuning-20261011-r3` añade la referencia humana,
la salida base anterior al ajuste y la ajustada en tres columnas. Permite
comparar los tres textos o cada par, elegir cualquiera como guía de alineación
y filtrar fragmentos distintos, con filas y desplazamiento compartidos.

- Cobertura y resultados idénticos a r2: 50 documentos base y 300 pares
  base/ajustado comprobados por página, partición, referencia y política.
- Sólo cambian los tres assets de interfaz y la fecha de construcción del
  catálogo; 57.443 payloads conservan exactamente sus hashes anteriores.
- `make verify`: 227 tests, compilación, lint y formato correctos, incluidos
  cinco controles Node de alineación y asociación de baseline.
- Doctor: 15 comprobaciones correctas, cero advertencias o fallos; dry-run
  correcto, sin escritura remota.
- Navegador: 43 controles sobre el bundle real local y la URL pública, y 46
  con fixtures, sin errores;
  escritorio y móvil, las tres guías y los tres pares, texto íntegro, baseline
  estable entre épocas y comparación provisional de la página test fallida.

Al cerrar r3, los punteros privados de candidata y release activa registraban
`ACTIVE_VERIFIED` y conservaban la propuesta, aprobación exacta y recibos de
activación y verificación. El puntero anterior de r2 queda archivado junto a la operación;
sus fuentes y resultados se conservan. No editar los bytes del bundle aprobado.

## Interfaz r4

- Release: `qwen-berrutti-finetuning-20261012-r4`.
- SHA-256 del manifest: `8391b6c7614846b0cd034dcbf18bc1376f26bbff3a1f0c28c84c98bb2f014108`.
- Mismo destino público y configuración; 57.448 archivos, 7.406.418.382 bytes.
- Imagen a la izquierda; selector para las tres transcripciones, cualquier par
  o una sola. Debajo, un panel único con referencia y texto comparado
  seleccionables, intercambio de orden y filtro de diferencias.
- Textos de introducción y ayuda reducidos. Se conservan los estados
  incompletos y la advertencia de contaminación del experimento cuatro.
- 57.443 payloads conservan sus hashes de r3. Sólo cambian los tres assets de
  interfaz y la fecha del catálogo; datos, resultados y puntuaciones idénticos.
- `make verify`: 232 tests, compilación, lint y formato correctos; seis pruebas
  Node de conservación de caracteres, pares y asociación de baseline.
- Navegador: 73 controles con el bundle real y la URL pública, y 76 con fixtures, sin errores,
  en escritorio y móvil. Comprueba siete vistas, seis parejas dirigidas,
  reconstrucción íntegra de ambos textos, ausencia de desbordamiento,
  provisionalidad de la pareja seleccionada y texto HTML literal.
- Doctor: 15 comprobaciones correctas, cero advertencias o fallos. Dry-run
  correcto. Tras la aprobación exacta, el CLI activó el mismo bundle y hash
  con salida cero y sin advertencias.

Los punteros activo y de candidata registran `ACTIVE_VERIFIED` para r4. La
operación conserva la aprobación exacta, el puntero anterior de r3 y todos los
recibos. La comparación independiente del árbol remoto completo no encontró
diferencias; el manifest público y los cuatro archivos de entrada coinciden
en tamaño y SHA-256. No reconstruir ni modificar el bundle aprobado.
