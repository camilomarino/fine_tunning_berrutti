# Interfaz de resultados Qwen–Berrutti

Viewer estático, sin dependencias externas. `results_viewer.py` copia estos tres
assets a un bundle privado nuevo junto con su catálogo, imágenes, fragmentos y
manifiesto. Abrir el bundle por HTTP; el catálogo y los detalles se cargan con
`fetch`, sin incluir un índice global de transcripciones.

La interfaz distingue entrenamiento, validación y test. Train muestra imágenes,
referencias humanas, recortes de línea y pérdidas registradas por época; no
inventa OCR ni CER de entrenamiento. Validación y test muestran página completa,
referencia, OCR base previo al ajuste y OCR del checkpoint seleccionado. La línea
base compartida es una evaluación previa de los tres brazos, con cero
actualizaciones; se carga para la misma página y partición al cambiar de época.

El selector de transcripciones permite ver las tres, cualquiera por separado o
cualquiera de los tres pares. La imagen permanece a la izquierda en escritorio;
en móvil, los paneles se apilan. La selección se conserva al cambiar de página
y se recupera al salir de train, donde sólo hay referencia humana.

Debajo hay un único panel de diferencias. Sus selectores independientes eligen
la referencia y el texto comparado entre humana, base y ajustada; un botón
intercambia sus posiciones. Las omisiones se tachan en rojo y las adiciones se
resaltan en verde. Se pueden ocultar los fragmentos iguales. La alineación
conserva cada carácter, incluidos los espacios, y acota la matriz, pasando a
líneas o bloques completos para textos muy largos. La granularidad efectiva se
muestra siempre. No recalcula CER ni altera textos.

Seleccionar la evaluación base nunca la presenta como una salida ajustada.
Una transcripción ausente permanece ausente; train no inventa OCR y la cuarta
prueba sólo admite su propia baseline. Las generaciones incompletas mantienen
su estado provisional en las transcripciones y en las parejas que las incluyen.
Una pareja completa no se marca provisional por el fallo del tercer texto.

El cuarto modelo se presenta por separado como extensión exploratoria, con
contaminación conocida del test. Sus composiciones de líneas Berrutti y LUISA se
filtran por época y una/dos columnas. La preparación de imágenes no implica que
el entrenamiento esté lanzado. Un modelo `PLANNED` conserva CER ausente y la
extensión nunca toma prestadas predicciones de la línea base compartida.

Las tarjetas describen brevemente las cuatro recetas y sus épocas. En una
extensión activa, `progress` contiene `epoch`, `images_completed`, `images_total`
y `phase`, calculados desde el checkpoint confirmado y la preparación sellada.
El porcentaje cuenta imágenes de la época, sin presentar páginas de origen como
si fueran todos los ejemplos. `optimizer_steps` incluye las actualizaciones
observadas antes del cierre final. La última validación disponible puede ser
un control intermedio y se identifica como tal; `selection` permanece ausente
hasta completar los cuatro cierres. La baseline visible cambia con el método.

## Contrato de lectura

- `catalog.json`, versión 1: métodos, páginas, evaluaciones y contadores. Las
  rutas de imágenes y fragmentos son relativas al bundle. `observed_at` identifica
  la evidencia y `generated_at` la fecha de publicación, sin presentar la web
  estática como un monitor en vivo.
- Cada método expone estado, épocas, actualizaciones, selección por validación,
  test y `training_history.epochs`. Un CER es un cociente, incluso si excede uno;
  la web lo multiplica por cien. `null` se muestra como ausencia de dato.
- `evaluations[].documents` enlaza por `page_id` al detalle de OCR. Las páginas
  y sus líneas conservan referencias originales y normalizadas por separado.
  `all_pages_completed=false` mantiene el agregado provisional; el detalle
  conserva fallo, reintento y motivos de los dos intentos como máximo.
- `synthetic_index` y `luisa_index` son opcionales. Se cargan sólo al elegir esa
  familia; las transcripciones de cada imagen se leen al abrir su detalle.

Todos los textos se insertan con `textContent`; el OCR no se interpreta como
HTML. Los recursos deben pertenecer al mismo directorio publicado. El diff
visual usa una alineación acotada por palabras (o bloques para textos muy
largos) y no recalcula ni sustituye el CER publicado.

La publicación usa un destino nuevo y el flujo transaccional del repositorio
`berrutti_benchmark`; las interfaces históricas existentes permanecen intactas.
No guardar imágenes, referencias, OCR ni bundles materializados en Git.
