# Operación técnica de los datos

[Volver a la guía de los tres conjuntos](README.md).

La [CLI común](CLI.md) ofrece exploración visual y consultas JSON por ejemplo
sin recorrer las carpetas de datos. Este documento conserva la operación de
importaciones, versiones e integridad; sus comandos también aceptan la entrada
`berrutti-data` en lugar de `python -m data`.

Esta guía describe los índices, la importación y la verificación. Para encontrar
imágenes o transcripciones, empezar por la guía anterior. Los comandos de
consulta se ejecutan desde la raíz del repositorio, con `.venv` activado.

## Catálogo, ubicaciones e índices

`catalog.json` describe las fuentes y sus identificadores estables:
`berrutti_175`, `luisa_tesseract_lines` y `archive_berrutti`.
`locations.local.json` registra las ubicaciones privadas de esta instalación.
Los archivos `remote.json` dentro de cada carpeta conservan su ubicación remota.

```bash
python -m data list --json
python -m data show luisa_tesseract_lines
python -m data path berrutti_175 --where local --kind raw
python -m data path berrutti_175 --where ruche --kind root
python -m data path luisa_tesseract_lines --where ruche --kind records
python -m data path luisa_tesseract_lines --where ruche --kind manifest
```

| `--kind` | Resultado |
| --- | --- |
| `raw` (predeterminado) | Carpeta de originales; en Archive, raíz de la biblioteca |
| `root` | Carpeta de la versión completa; en Archive, raíz de la biblioteca |
| `records` | Índice `records.jsonl` de Berrutti o LUISA |
| `manifest` | Resumen `release.json` de Berrutti o LUISA |

`list` y `show` incluyen el host; `path` devuelve solo la ruta en ese host.
Son lecturas de metadatos locales, sin conexión ni comprobación en vivo del
almacenamiento. Un destino o tipo de archivo no registrado falla explícitamente.
Un clon sin `locations.local.json` muestra el catálogo sin inventar ubicaciones.
`BERRUTTI_DATA_LOCATIONS` permite elegir otro registro privado.

Para el estado vivo y los archivos de Archive, usar su interfaz específica:

```bash
python -m archive_berrutti agent list --state complete
python -m archive_berrutti agent info --rolls 099
python -m archive_berrutti agent files --rolls 099 --kind image --limit 10
```

Estas consultas leen el índice remoto por SSH sin reservar CPU. La documentación
de [Archive para agentes](../archive_berrutti/README.md#uso-por-agentes) cubre
descarga, procesamiento, trabajos en segundo plano y consulta por ID.

## Versiones de Berrutti y LUISA

```text
<conjunto>/
  current -> releases/<version>/
  releases/<version>/
    raw/                  # originales conservados
    records.jsonl         # una línea JSON por ejemplo
    files.jsonl           # inventario de tamaños y hashes SHA-256
    release.json          # procedencia, cantidades y hash del inventario
    split-v1.json         # solo Berrutti: partición vigente sin cambios
```

`records.jsonl` relaciona cada ejemplo con su imagen y texto: `id`, `dataset`,
`unit`, `image`, `reference` e `image_sha256`. Las rutas son relativas a la
carpeta que contiene `release.json`, no al directorio de ejecución. El índice
de LUISA agrega los textos, la predicción OCR y los datos de su partición de
origen. Estos índices contienen datos y también permanecen fuera de Git.

`files.jsonl` inventaría originales e índices derivados; `release.json` guarda
su SHA-256, totales y procedencia. Berrutti conserva el manifiesto original,
licencias y anotaciones. LUISA verifica el árbol Git completo del subdirectorio
fijado, la correspondencia por ID entre CSV, imagen, transcripción y OCR, y
decodifica cada imagen. `image_format` registra el formato real de la imagen,
independientemente de la extensión. Ver la [ficha LUISA](luisa_tesseract_lines/README.md)
para las diferencias de espaciado conservadas entre sus referencias.

`current` facilita explorar. Para una corrida nueva, **guardar la ruta resuelta
de la versión y sus hashes en la configuración**. Un cambio futuro del enlace
no debe cambiar los datos de una corrida ya configurada.

## Importar o verificar otra instalación

Estos comandos realizan trabajo sobre los archivos. En Ruche se ejecutan en
una asignación CPU de Slurm; las consultas `list`, `show` y `path` anteriores
no necesitan esa asignación.

```bash
# En esta máquina: importar una copia Berrutti v3, conservando la fuente.
python -m data import-berrutti --source /ruta/berrutti-v3 --store data

# En esta máquina: verificar todos los archivos de la versión local.
python -m data verify data/berrutti_175/current

# Dentro de una asignación CPU de Ruche, con código y entorno preparados:
python -m data import-luisa --archive /ruta/luisa.tar.gz \
  --store "$WORKDIR/berrutti-data"
```

La URL fijada de LUISA está en su [ficha](luisa_tesseract_lines/README.md#repetir-la-importación).
[import.sbatch](import.sbatch) permite ejecutar importación o verificación:
preparar una copia nueva del código en WORKDIR, definir `BERRUTTI_DATA_CODE` y
`BERRUTTI_DATA_PYTHON` con el entorno existente, comprobar cuenta, partición y
cuota, y hacer `sbatch --test-only` antes del envío real. No ejecutar Python,
hashes, escaneos o descompresión en el login. Los imports no instalan dependencias.

La importación escribe primero en `.staging/<id>/`, verifica y publica la
carpeta completa mediante un renombrado. Una versión existente se rechaza; no
se sobrescribe. Si falla, no cambia `current` y conserva el intento parcial para
diagnóstico. `verify` comprueba todos los hashes y rechaza archivos adicionales.

## Compatibilidad y almacenamiento

En Ruche los tres conjuntos están bajo `$WORKDIR/berrutti-data/`:
`berrutti_175/`, `luisa_tesseract_lines/` y `archive/library-v1/`.
`operations/<id>/` conserva código fijado y registros de las importaciones.

La copia anterior del benchmark y las copias usadas por corridas de Ruche
permanecen en sus rutas. `.private/dataset` apunta a la nueva copia local de
`data/berrutti_175/current/raw/`. Los recortes, OCR auxiliar y resultados de
entrenamiento siguen siendo derivados de cada corrida, fuera de estas fuentes.
Archive conserva su ruta física y su estructura, porque las solicitudes y
trabajos existentes contienen rutas absolutas.

Esta organización no cambia las particiones ni incorpora LUISA al entrenamiento:
hay solapamiento detectado con páginas del test de Berrutti. En LUISA,
`source_split` conserva el `train`/`test` de origen, `project_split` queda sin
asignar y `requires_overlap_audit` indica que falta resolver la identidad de
origen. Consultar la política de cada conjunto antes de definir una mezcla nueva.
