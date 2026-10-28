# Los tres conjuntos de datos

Este es el código de datos: acceso, integridad y preparación CPU. No importa
entrenadores ni herramientas de publicación. Las preparaciones específicas
están en [preparation/](preparation/README.md); el split Berrutti está en
[berrutti_175/split-v1.json](berrutti_175/split-v1.json).

El proyecto cuenta con **páginas Berrutti transcritas, líneas LUISA transcritas y
rollos de Archive sin transcribir**. Su almacenamiento principal está en
**Ruche**. Esta carpeta del repositorio reúne la documentación y el catálogo
para encontrarlos; no contiene una copia local de todos los datos.

**Entrada única: `berrutti-data` (o `python -m data`).** Abre un menú con los
tres conjuntos y navegación con flechas. Para agentes, `berrutti-data agent list`
describe todas las fuentes. Ver [cómo usar la CLI común](CLI.md).

## Qué contiene cada conjunto

| Conjunto | Qué hay en cada ejemplo | Cantidad y disponibilidad | Para qué sirve |
| --- | --- | --- | --- |
| [Berrutti 175](berrutti_175/README.md) | Imagen y transcripción humana de página completa; **8.975 cajas de líneas con su texto humano** | **175 páginas**, completas en esta máquina y en Ruche | Entrenamiento y evaluación actuales de OCR de páginas |
| [LUISA / Tesseract](luisa_tesseract_lines/README.md) | Una imagen recortada de una línea, su transcripción humana y el texto que predijo Tesseract | **23.774 líneas**, completas en Ruche; sin copia local completa | Material adicional para estudiar OCR de líneas; todavía separado del entrenamiento actual |
| [Archive Berrutti](archive_berrutti/README.md) | Rollos que se descargan como ZIP, contienen PDF y se convierten en imágenes de páginas | **Descarga por rollos**; hay bibliotecas parciales en Ruche y en esta máquina | Gran colección de imágenes sin transcripción humana asociada |

**Una página, una línea y un rollo son unidades distintas.** En Berrutti 175 se
cuentan páginas, no necesariamente 175 documentos históricos independientes.
LUISA contiene recortes de líneas. En Archive, un rollo puede tener varios ZIP
y PDF, y muchas páginas finales. Los recortes creados por nuestros experimentos
sobre Berrutti 175 siguen siendo derivados de Berrutti; no son el conjunto LUISA.

OCR significa reconocimiento automático de texto. El texto de Tesseract es una
predicción automática y puede contener errores; se guarda separado de la
transcripción humana.

## Dónde están los archivos

Las rutas de esta máquina parten de la raíz del repositorio. Las rutas de Ruche
se abren **en Ruche**: `$WORKDIR` representa allí el espacio de trabajo del usuario.

| Conjunto | En esta máquina | En Ruche |
| --- | --- | --- |
| Berrutti 175 | `data/berrutti_175/current/raw/` | `$WORKDIR/berrutti-data/berrutti_175/current/raw/` |
| LUISA / Tesseract | Solo documentación y ubicación remota | `$WORKDIR/berrutti-data/luisa_tesseract_lines/current/raw/` |
| Archive Berrutti | `data/archive_berrutti/local/` | `$WORKDIR/berrutti-data/archive/library-v1/` |

`current` es un acceso directo a la versión importada. `raw` guarda los archivos
originales. Los datos reales de Berrutti y LUISA están en carpetas `releases/`
identificadas por versión. Archive conserva su propia biblioteca por rollos.
**Las bibliotecas local y remota de Archive son independientes**: descargar en
una no copia los archivos a la otra.

Para ver las **rutas absolutas de esta instalación**, ejecutar desde la raíz
del repositorio, en esta máquina:

```bash
source .venv/bin/activate
python -m data list
```

Para obtener una ruta concreta:

```bash
python -m data path berrutti_175 --where local
python -m data path berrutti_175 --where ruche
python -m data path luisa_tesseract_lines --where ruche
python -m data path archive_berrutti --where ruche
```

Estos comandos consultan ubicaciones registradas: **no descargan, no reservan
CPU y no montan carpetas remotas**. Una ruta de Ruche no se puede abrir como si
fuera local. En esta instalación, omitir `--where` elige Ruche.

## Quiero ver las imágenes o leer el texto

| Conjunto | Dentro de la ruta indicada arriba |
| --- | --- |
| Berrutti 175 | `images/` contiene las páginas; `full_text_transcription/`, las transcripciones; `bounding_boxes_lines/`, las cajas que ubican las líneas |
| LUISA / Tesseract | `images/` contiene cada imagen `ID.gif`, su texto humano `ID.gt.txt` y el OCR `ID.pred.txt`; los CSV de la carpeta `raw/` reúnen esos textos por ID |
| Archive Berrutti | El explorador muestra las rutas de ZIP, PDF e imágenes finales de cada rollo; son subcarpetas de `rolls/` |

Las imágenes de la versión importada de LUISA **contienen PNG aunque se llamen
`.gif`**. Se mantienen sus nombres originales; los lectores que detectan el
formato por contenido, como Pillow, pueden abrirlas.

Para explorar cualquiera de los tres conjuntos desde esta máquina:

```bash
python -m data           # menú de conjuntos en Ruche
python -m data --local   # menú de conjuntos en esta máquina
```

Usar ↑/↓ para elegir un conjunto y un ejemplo, Enter o → para abrirlo y ← para
volver. Berrutti permite ver imágenes, texto y líneas con cajas; LUISA muestra
la imagen de línea, su texto humano y su OCR. Archive conserva imágenes, PDF,
ZIP y acciones por rollo, con sus rutas. **Un rollo finalizado
tiene todas sus imágenes finales**; aparecer en el catálogo no significa que
esté descargado. El estado y el número de imágenes se consultan en el explorador.

## Qué se usa para entrenar

Berrutti 175 conserva **125 páginas de entrenamiento, 25 de validación y 25 de
test**, agrupadas por rollo. LUISA conserva sus grupos publicados de 21.611
líneas de entrenamiento y 2.163 de test, que son particiones de otro conjunto.
**Se detectó material compartido entre LUISA y el test de Berrutti**: LUISA no
se incorpora automáticamente al entrenamiento. Archive aporta imágenes sin
etiquetas. Registrar una fuente en el catálogo no la agrega a ninguna corrida.

## Más información y uso por agentes

- Las fichas de [Berrutti 175](berrutti_175/README.md),
  [LUISA](luisa_tesseract_lines/README.md) y
  [Archive](archive_berrutti/README.md) explican su origen y sus archivos.
- La [guía técnica](OPERACION.md) explica índices, versiones, importación,
  verificación y compatibilidad con las rutas anteriores.
- La [CLI común para personas y agentes](CLI.md) permite explorar los tres
  conjuntos, consultar ejemplos, texto, cajas y archivos, y gestionar Archive.
  `python -m data list/show/path` conserva sus consultas rápidas anteriores.

Git guarda las guías, el código y [catalog.json](catalog.json), que describe
las tres fuentes. Los archivos de datos y `locations.local.json`, que contiene
las rutas privadas de esta instalación, quedan fuera de Git. **Clonar el
repositorio no descarga los datasets ni configura sus ubicaciones.**
