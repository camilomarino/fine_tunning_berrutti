# Web de resultados Qwen–Berrutti

La web muestra las tres estrategias Berrutti y la extensión Berrutti–LUISA
en una publicación independiente del benchmark. Incluye imágenes y referencias
humanas de las 125 páginas train, 25 de validación y 25 de test, los 5.212
recortes de entrenamiento y las predicciones ya guardadas. Si se incorpora la
preparación del experimento cuatro, muestra todas sus páginas sintéticas y
líneas LUISA, conservando las particiones originales de LUISA.

Las cuatro tarjetas resumen los datos y las épocas de cada variante. La
extensión muestra las actualizaciones observadas, el porcentaje confirmado
por imágenes de la época en curso y la última validación cerrada. Los controles
intermedios se ordenan por avance real y nunca se presentan como selección
final. Al elegir la extensión se muestra su baseline propia, con su política
de generación; los tres brazos originales conservan la baseline compartida.
La fecha identifica una publicación estática consultada manualmente: no se
reactivan informes periódicos, vigilancia ni llamadas al modelo.

Entrenamiento muestra muestras supervisadas y pérdidas disponibles, sin inventar
OCR ni CER de train. Validación permite recorrer las épocas; test muestra sólo
el checkpoint seleccionado. La línea base compartida es una evaluación previa
con cero actualizaciones. Una generación incompleta y sus dos intentos se
conservan visibles; su CER sigue siendo provisional. El cuarto experimento se
identifica como exploratorio, con contaminación conocida del test, incluso si
todavía no tiene resultados.

Validación y test permiten elegir una, dos o las tres transcripciones: referencia
humana, base anterior al fine-tuning y ajustada de la época elegida. La imagen
permanece a la izquierda en escritorio. Debajo hay un único panel de comparación,
con selectores de referencia y texto comparado, un botón para intercambiarlos y
una opción para mostrar sólo diferencias. Admite los tres pares en ambas
direcciones y resalta omisiones y adiciones dentro del mismo texto.
Conserva originales y normalizados, identifica la granularidad visual y no
sustituye el CER publicado. Sólo las parejas que incluyen una generación
incompleta son provisionales. La baseline corresponde a la misma página y
partición; la extensión requiere su baseline propia. Train conserva sólo las
referencias, porque no existe OCR evaluado de entrenamiento.

## Construcción desde evidencia cerrada

```bash
.venv/bin/python -m reporting.berrutti.results_viewer \
  --dataset /ruta/privada/berrutti/raw \
  --split data/berrutti_175/split-v1.json \
  --cohort /ruta/privada/inputs/cohort.json \
  --evidence /ruta/privada/evidence \
  --snapshot /ruta/privada/snapshot.json \
  --mix-data /ruta/privada/mix-data \
  --output /ruta/privada/nuevo-bundle
```

`--mix-data` es opcional. Consume el manifest, recibo cerrado, imágenes
sintéticas y caché de crops de la preparación verificada. También se puede
aportar `--luisa-root` para usar los originales LUISA. Nunca se genera OCR ni
se modifica un entrenamiento al construir la web.

`evidence/` contiene `shared_baseline/`, `pages/`, `lines/`, `combined/` y,
cuando exista, `experiment4/`. Cada directorio conserva las evaluaciones
`evaluations/epoch-NN-validation/` y `epoch-NN-test/`, sus `closed.json` y los
recibos por página e intento. No hacen falta checkpoints, entornos ni credenciales.
Se exportan sólo campos públicos permitidos, imágenes verificadas y textos;
los eventos y las rutas operativas no se copian al bundle.

La salida es create-only y contiene fragmentos con nombres SHA-256, un catálogo
sin corpus de textos embebido y `manifest.json` con todos los tamaños y hashes.
Un bundle cerrado no se edita: para actualizar resultados se construye otro.
Los datos y artefactos generados deben permanecer fuera de Git.

## Vista previa

```bash
.venv/bin/python -m http.server 8766 --bind 127.0.0.1 \
  --directory /ruta/privada/nuevo-bundle
```

Abrir `http://127.0.0.1:8766/`. La web es estática y carga sólo los documentos
solicitados. No necesita backend, CDN, servicio de modelos ni acceso al cluster.

## Preparación para Ampere

Se reutiliza el CLI transaccional de `berrutti_benchmark` y su guía
`skills/deploy-berrutti-ocr-viewer/SKILL.md`. El destino separado se declara en
`configs/publish/ampere-finetuning.toml` y corresponde a:

`https://iie.fing.edu.uy/~cmarino/ocr/qwen_berrutti_finetuning/`

Desde el checkout de fine-tuning, con paths absolutos para los artefactos:

```bash
BENCHMARK=/ruta/al/berrutti_benchmark
BUNDLE=/ruta/privada/nuevo-bundle
PUBLISH_CONFIG="$PWD/configs/publish/ampere-finetuning.toml"
RELEASE_ID=qwen-berrutti-finetuning-YYYYMMDD-r1

"$BENCHMARK/.venv/bin/berrutti-bench" --project-root "$BENCHMARK" doctor \
  --profile publish --dataset configs/datasets/berrutti-v3.toml \
  --bundle "$BUNDLE" --warnings-as-errors

"$BENCHMARK/.venv/bin/berrutti-bench" --project-root "$BENCHMARK" publish \
  --bundle "$BUNDLE" --config "$PUBLISH_CONFIG" --host ampere \
  --destination /iie/home01/cmarino/webdocs/ocr/qwen_berrutti_finetuning \
  --keep-releases 2 --release-id "$RELEASE_ID"
```

El segundo comando es un dry-run local y no publica. La activación exige
aprobación explícita del destino, release ID y SHA-256 exactos del manifest.
Después se repite exactamente el plan aprobado con
`--expected-manifest-sha256 SHA_APROBADO --activate`. Toda escritura remota
pasa por ese CLI; nunca se sobreescriben los viewers congelados v2/v4.

Comprobar después el hash del manifest público y la igualdad del árbol mediante
el procedimiento de la guía. Conservar los recibos privados del doctor,
dry-run, activación y verificación junto a la operación local.
