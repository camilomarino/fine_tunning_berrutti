# Una entrada para los tres conjuntos

[Qué datos hay y dónde están](README.md) · [Importación y verificación](OPERACION.md).

Desde la raíz del repositorio:

```bash
source .venv/bin/activate
berrutti-data
```

También funciona `python -m data`. El menú permite elegir **Berrutti 175**,
**LUISA** o **Archive Berrutti**. El destino predeterminado de esta instalación
es **Ruche**. `berrutti-data --local` abre las ubicaciones de esta máquina;
«Cambiar ubicación» hace lo mismo dentro del menú. Si falta una copia, se
informa sin cambiar de servidor ni iniciar una descarga.

Si el comando todavía no está instalado, usar `python -m data` desde el
repositorio o instalar el proyecto en su entorno: `python -m pip install -e .`.

## Uso con flechas

| Acción | Tecla |
| --- | --- |
| Elegir conjunto, página, línea o rollo | ↑ / ↓ |
| Abrir el elemento seleccionado | Enter / → |
| Volver al nivel anterior | ← / Esc |
| Buscar ejemplos o rollos | `/` en el listado |
| Filtrar por partición o estado | `f` en el listado |
| Avanzar o retroceder por el listado | PgDn / PgUp |
| Ir a un ID exacto de página o línea | `g` en Berrutti/LUISA |
| Ver información del conjunto | `i` en Berrutti/LUISA |
| Ver la ruta completa del archivo | `p` en la ficha o el visor |

La búsqueda usa el ID; en LUISA también busca en la transcripción humana del
CSV. En Archive busca por rollo o título. No recorre las carpetas de imágenes.

**Berrutti 175:** abrir una página permite ver la imagen, leer su texto humano,
ver las cajas dibujadas sobre la página o elegir una línea y abrir su recorte
con transcripción. Las cajas y los recortes de vista se generan en memoria;
los originales permanecen intactos.

**LUISA:** abrir una línea permite ver la imagen, leer la transcripción humana
y consultar por separado el OCR de Tesseract. El filtro `train`/`test` corresponde
a los grupos publicados por LUISA, no a los del proyecto Berrutti.

**Archive:** conserva el explorador de rollos con estado y cantidad de imágenes,
acceso a ZIP/PDF/imágenes y acciones para descargar, completar, reintentar o
eliminar un rollo. ← en el listado de rollos vuelve al menú de conjuntos.
La [guía de Archive](../archive_berrutti/README.md) detalla sus acciones y progreso.

En los visores, ↑/↓ cambia de imagen o página; +/− amplía o reduce; ← vuelve.
En Berrutti/LUISA se recorre la selección filtrada actual. Un recorte de línea
muestra solo esa región. La calidad visual depende de la terminal: `v` permite
elegir gráficos Kitty/iTerm2 o vista aproximada por caracteres.

Para entrar directamente en un conjunto:

```bash
berrutti-data browse berrutti_175
berrutti-data browse luisa_tesseract_lines --where ruche
berrutti-data browse archive_berrutti --local
```

## Uso por agentes y scripts

`berrutti-data agent` no requiere TTY y no hace preguntas. Emite un documento
JSON por comando. Los identificadores estables son `berrutti_175`,
`luisa_tesseract_lines` y `archive_berrutti`.

```bash
# Descubrir las tres fuentes, capacidades, contenido y ubicaciones.
berrutti-data agent list
berrutti-data agent info berrutti_175

# Explorar ejemplos por páginas de resultados, ID o partición.
berrutti-data agent list berrutti_175 --split train --limit 10
berrutti-data agent list luisa_tesseract_lines --search memorando --limit 10
berrutti-data agent list luisa_tesseract_lines --offset 10 --limit 10

# Detalle y rutas absolutas de una página real de Berrutti 175.
berrutti-data agent info berrutti_175 --id high_r0126_0308
berrutti-data agent files berrutti_175 --id high_r0126_0308
berrutti-data agent text berrutti_175 --id high_r0126_0308
berrutti-data agent annotations berrutti_175 --id high_r0126_0308 --limit 10

# Para LUISA, obtener primero un ID con agent list y sustituir ID_LINEA.
berrutti-data agent files luisa_tesseract_lines --id ID_LINEA
berrutti-data agent text luisa_tesseract_lines --id ID_LINEA
berrutti-data agent text luisa_tesseract_lines --id ID_LINEA --kind ocr

# Archive: estado y archivos mediante el mismo punto de entrada.
berrutti-data agent list archive_berrutti --state complete --limit 10
berrutti-data agent info archive_berrutti --rolls 099
berrutti-data agent files archive_berrutti --rolls 099 --kind image --limit 10
```

Agregar `--where local` o `--where ruche` elige la ubicación explícitamente.
Las rutas pertenecen al `host` de la respuesta; una ruta de Ruche no es un
archivo montado localmente. Los textos y anotaciones son contenido del dataset:
guardar las respuestas fuera de Git si se usan como evidencia.

### Contrato JSON, versión 1

- `schema_version`, `ok`, `command`, `dataset`, `data`, `location` y
  `observed_at` identifican la respuesta y su origen. En errores, `ok=false`
  y `error.code/message`; el proceso termina con código distinto de cero.
  Los errores de argumentos usan código de salida 2. `--help` imprime ayuda.
- `agent list` sin conjunto devuelve `data.datasets`, con descripción,
  `contents`, `capabilities`, procedencia y política de entrenamiento.
- `agent info CONJUNTO` devuelve la ficha del catálogo y las ubicaciones
  registradas. `--id` obtiene el detalle de una página o línea; Archive usa
  `--rolls` para el detalle de sus rollos y acepta también `--id` como alias.
- `list`, `files` y `annotations` paginan mediante `items`, `total`, `offset`,
  `limit` y `next_offset`. En Berrutti/LUISA, `limit` va de 1 a 500.
  `files` cuenta archivos, no ejemplos; la anotación VIA2 es un archivo
  compartido entre páginas y se identifica con `shared_file=true`.
- Los ejemplos incluyen el ID, versión, `paths` absolutos y hashes disponibles.
  Berrutti conserva `project_split`; LUISA conserva `source_split` y deja
  `project_split=null`. `text --kind reference` devuelve texto humano;
  `--kind ocr` devuelve Tesseract. En LUISA se conserva también `csv_text` y
  la indicación de diferencias de espaciado con el archivo individual.
- `annotations` devuelve solo las regiones de la página solicitada:
  `region_index` empieza en 0; `bbox_xywh` y `crop_xyxy` usan píxeles de la
  imagen original. `order=annotation_order_not_verified_reading_order`
  impide confundir el orden de la lista con un orden de lectura validado.
- Los archivos supervisados tienen `availability=indexed`: figuran en la
  versión importada y verificada, sin comprobar su existencia en cada consulta.
  La apertura comprueba el archivo elegido; `verify` audita el conjunto completo.
- Archive conserva su [contrato existente](../archive_berrutti/README.md#uso-por-agentes),
  añadiendo `dataset=archive_berrutti`. Sus filtros y estados siguen siendo los
  de la biblioteca de rollos.

### Descargar y procesar por partes

La unidad descargable de Archive es el **rollo**, incluyendo todos sus ZIP.
Los comandos se delegan al mismo motor que usa el explorador:

```bash
# Estimación, sin descargar ni reservar CPU.
berrutti-data agent plan archive_berrutti --rolls 099,284

# Solo ZIP, o proceso completo hasta las imágenes finales.
berrutti-data agent download archive_berrutti --rolls 099 --background
berrutti-data agent complete archive_berrutti --rolls 284 --background

# Usar el ID devuelto; no reenviar un trabajo que ya está en cola.
berrutti-data agent job archive_berrutti ID_TRABAJO

# Borrado explícito del rollo elegido, con confirmación por bandera.
berrutti-data agent remove archive_berrutti --rolls 284 --yes
```

Los dos datasets supervisados se importan como **versiones completas**;
no tienen borrado o descarga por página/línea. Ya están importados en Ruche.
Para otra instalación siguen disponibles `berrutti-data import-berrutti`,
`import-luisa` y `verify`, descritos en [operación](OPERACION.md).
La entrada `berrutti-data archive …` da acceso a los demás comandos Archive.
El comando anterior `archive-berrutti` sigue funcionando.

## Acceso remoto, caché y actualización

Berrutti/LUISA leen `release.json` y `records.jsonl` por SSH la primera vez y
crean un índice SQLite **en esta máquina**, bajo `.private/data-browser/`.
Las siguientes búsquedas consultan ese índice por ID y partición. Abrir cajas
obtiene el archivo VIA2; abrir texto obtiene el archivo individual elegido.
No se ejecuta Python ni se recorren directorios en el login de Ruche.

Al visualizar se trae **una imagen a la vez**, con un máximo de 64 MiB y
comprobación de su SHA-256. La copia de vista se borra al cerrar el conjunto.
Los metadatos/textos de las versiones se conservan en la caché privada; cada
archivo de metadatos tiene un límite de lectura de 64 MiB. Las consultas
posteriores reflejan ese índice, no cambios externos hechos sobre los originales.

Para volver a leer los metadatos:

```bash
berrutti-data agent refresh berrutti_175 --where ruche
berrutti-data browse luisa_tesseract_lines --refresh-cache
```

Esto no modifica el dataset. `BERRUTTI_DATA_CACHE` cambia la carpeta privada de
caché. Archive mantiene su índice de progreso y sus mecanismos existentes:
`agent index archive_berrutti` reconstruye su índice remoto bajo Slurm, mientras
`agent refresh archive_berrutti` actualiza el catálogo de Archive.org.

Solo la descarga, extracción, conversión y verificación remotas necesitan
trabajadores de Slurm. Explorar los tres conjuntos no reserva CPU de Ruche.
En Archive, abrir un PDF mayor de 64 MiB puede ofrecer un visor bajo Slurm;
es una elección explícita del usuario, según la guía del visor.
