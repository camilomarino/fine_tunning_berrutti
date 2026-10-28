# LUISA / Tesseract — 23.774 líneas

[Volver a los tres conjuntos](../README.md).

Contiene **23.774 imágenes de líneas de texto**, cada una con su transcripción
humana de LUISA y el texto que predijo Tesseract. La transcripción humana sirve
como referencia; el OCR de Tesseract se conserva como información auxiliar.
Procede de [Memoria.uy / datasets-publicos](https://gitlab.fing.edu.uy/memoria/datasets-publicos/-/tree/main/luisa_tesseract_lines).

**Está descargado y verificado en Ruche**, sin copia local completa. Se mantiene
separado del entrenamiento actual porque comparte material con el test de
Berrutti. No son los recortes que generan nuestros experimentos sobre las 175
páginas: es una fuente publicada por separado.

## Dónde abrir las imágenes y los textos

En Ruche, abrir `$WORKDIR/berrutti-data/luisa_tesseract_lines/current/raw/`.
Dentro de `images/`, los archivos se relacionan por el mismo identificador:

| Nombre de archivo | Contenido |
| --- | --- |
| `ID.gif` | Imagen de una línea; el nombre termina en `.gif`, pero su contenido es PNG |
| `ID.gt.txt` | Transcripción humana de esa línea |
| `ID.pred.txt` | Texto reconocido automáticamente por Tesseract |

`ID` representa un número de ocho dígitos. Los archivos CSV de `raw/` reúnen los
textos por ID: `gts…csv` contiene referencias humanas y `preds…csv` el OCR.
Hay **21.611 líneas en el grupo de entrenamiento publicado y 2.163 en el de
test**; esos grupos se conservan y no son las particiones del proyecto Berrutti.

Desde la raíz del repositorio en esta máquina, con `.venv` activado:

```bash
python -m data path luisa_tesseract_lines --where ruche
python -m data path luisa_tesseract_lines --where ruche --kind records
```

El primer comando devuelve la ruta remota de originales; el segundo, el índice
que relaciona imágenes y textos. Ninguno copia los archivos a esta máquina.
`remote.json` y `../locations.local.json` registran esa ubicación privada.

`berrutti-data browse luisa_tesseract_lines` abre el explorador con búsqueda por
ID o texto humano y filtros por partición de origen. La ficha separa imagen,
transcripción humana y predicción Tesseract. Para agentes, la misma entrada
ofrece `agent list/info/files/text luisa_tesseract_lines`; ver la [guía CLI](../CLI.md).
Esta fuente no publica cajas de las líneas sobre sus páginas originales ni
una correspondencia de página/documento que permita reconstruir su orden global.

## Estructura e índices

```text
current/
  raw/
    README.md                         # documentación original
    gts21611.csv, gts2163.csv          # columnas id, content; referencia humana
    preds21611.csv, preds2163.csv      # mismas columnas; Tesseract
    images/
      00000000.gif                    # original, sin reconversión
      00000000.gt.txt
      00000000.pred.txt
      ...
  records.jsonl                       # pares y procedencia, indexados por ID
  files.jsonl                         # inventario completo con SHA-256
  release.json
```

El importador cruza las tablas **por ID**, no por posición de fila; rechaza
duplicados, solapamiento entre particiones publicadas, archivos ausentes,
diferencias de contenido CSV/sidecar e imágenes no decodificables. Comprueba todos los bytes
originales contra el árbol Git fijado, no solo el nombre del archivo descargado.
Conserva los nombres y bytes originales. La verificación de la versión importada
confirmó que **las 23.774 imágenes son PNG con extensión `.gif`**.
`image_format` registra el formato real
detectado, y `release.json` resume las cantidades por formato. Un consumidor
puede abrirlos con Pillow y convertir a RGB en memoria, sin renombrar los originales.

La auditoría completa encontró **658 transcripciones del test publicado** con
diferencias de espaciado entre CSV y `.gt.txt`, sin diferencias de contenido al
colapsar espacios. Las 21.611 referencias de train y las 23.774 predicciones OCR
coinciden al quitar espacios extremos. Se conservan ambos originales:
`reference_text` proviene del CSV, `reference` apunta al sidecar, y
`csv_sidecar_agreement` indica `exact` o `whitespace_only`. Los totales se guardan
en `release.json`; diferencias de contenido real detienen la importación.

Cada fila normalizada agrega dimensiones, hash de imagen y rutas relativas.
`source_split` conserva `train` o `test`. `project_split`, `source_page` y
`source_document` son `null`: la publicación no proporciona esa correspondencia.
`requires_overlap_audit=true` hace explícito que falta resolverla.

## Repetir la importación

La copia actual usa la revisión Git
`9b9ccccddf0b1e68199bd6cc66c6e90a2434494c`, con árbol del subdirectorio
`519237ab13cc3322f263eba0cfa6344e11d6cb56`. La versión física se guarda en
`$WORKDIR/berrutti-data/luisa_tesseract_lines/releases/<revision>/`;
`current` apunta a ella. `python -m data show luisa_tesseract_lines` devuelve
la ficha y las ubicaciones en JSON.

Para repetir la descarga en una **asignación CPU**, usar el archivo del
subdirectorio y commit fijados, no clonar otros datasets del repositorio:

```bash
curl --fail --location --retry 2 --max-filesize 2147483648 \
  --output /ruta/nueva/luisa.tar.gz \
  'https://gitlab.fing.edu.uy/api/v4/projects/12840/repository/archive.tar.gz?sha=9b9ccccddf0b1e68199bd6cc66c6e90a2434494c&path=luisa_tesseract_lines'
python -m data import-luisa --archive /ruta/nueva/luisa.tar.gz \
  --store "$WORKDIR/berrutti-data"
```

El importador limita la extracción a 4 GiB y 80.000 archivos, rechaza enlaces y
rutas fuera de la carpeta solicitada y no publica una versión parcial.
Ver la [guía técnica](../OPERACION.md) para preparar código y entorno bajo Slurm
y para verificar una versión ya importada.

## Relación con Berrutti y condiciones de origen

Una inspección privada previa detectó líneas del `train` publicado de LUISA que
coinciden con páginas de **test de Berrutti**; varias parejas se comprobaron
visualmente. Esa inspección es preliminar, no un mapa exhaustivo de documentos.
La evidencia se conserva en `.private/inspections/20261008-luisa-lines-e2a1ca8a/`.
Por eso incorporar el conjunto al catálogo no lo agrega al entrenamiento actual.
Se necesita una auditoría de origen antes de evaluar una nueva mezcla.

No se encontró una licencia específica en el directorio ni en la raíz del
repositorio inspeccionados. Se conserva la documentación publicada y se registra
esa ausencia; la licencia de Berrutti 175 no se extiende automáticamente a LUISA.
