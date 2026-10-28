# Archive Berrutti

Archive es uno de los tres conjuntos del proyecto. Para ubicar también las
páginas Berrutti 175 y las líneas LUISA, ver la [guía de datos](../data/README.md).
La entrada común `berrutti-data` permite elegir los tres conjuntos;
`berrutti-data agent … archive_berrutti` delega las operaciones de esta biblioteca
al mismo motor. Ver la [CLI unificada](../data/CLI.md).

Herramienta independiente para inventariar y explorar el Archivo SID de
Archive.org, descargar todos los ZIP o lotes elegidos, extraer sus PDF y convertir
páginas a imágenes. El código vive en esta carpeta. **En esta máquina, el destino
predeterminado es Ruche**, en `$WORKDIR/berrutti-data/archive/library-v1`.
`--local` conserva el uso local y su biblioteca anterior bajo `.private/`, fuera
de Git. Los perfiles privados se consultan con `archive-berrutti profile show`.

Para mantener los datos en Ruche y controlarlos desde otra máquina, ver
[Biblioteca remota y perfiles de servidor](REMOTE_RUCHE.md). Describe conexión,
cuotas, configuración, actualización del código y ejecución CPU bajo Slurm.
El explorador y las consultas de lectura de `agent` leen un índice por SSH sin
reservar CPU de Ruche. Conserva la interfaz y los visores; las descargas se envían como
trabajos independientes y continúan al cerrar la sesión.

## Entrada interactiva

Desde la raíz del repositorio:

```bash
source .venv/bin/activate
archive-berrutti                   # biblioteca remota predeterminada
archive-berrutti --local           # biblioteca local
archive-berrutti --profile ruche   # servidor explícito
```

También funciona `python -m archive_berrutti` sin instalar el comando.
Un `--root` explícito sin `--profile` selecciona local, por compatibilidad con
scripts existentes. Con `--profile ruche`, las rutas son de Ruche. Si la conexión
falla, se informa el error sin pasar a local. Ver [perfiles](REMOTE_RUCHE.md#uso-diario).
**La entrada es un listado de todos los rollos con tres columnas:**
**Rollo**, **Imágenes** finales guardadas y **Estado**. El estado indica si está
listo, qué está haciendo o cuál es el próximo paso que falta. Por ejemplo:

```text
Rollo      Imágenes  Estado
099               0  Falta extraer PDF
284               0  Falta extraer PDF
321               0  Falta descargar
```

La cabecera explica las etapas, que se repiten por cada ZIP del rollo:
**1 Descarga → 2 Extraer PDF → 3 Crear imágenes**. Durante una conversión,
una fila puede mostrar `099 · 54/2052 · 3/3 Creando imágenes · ~28 min`.
La columna Imágenes muestra **hechas/total** cuando se conoce el total de
páginas del rollo. Si falta contar páginas de otros ZIP, el total del rollo
aparece como `?` y el estado muestra el contador del ZIP actual.

Cuando están todos los PDF extraídos, muestra **Falta convertir a imágenes**;
cuando están todas las imágenes finales, **Completo**. Un fallo muestra
**Error · reintentar**. La acción principal resuelve todos los pasos restantes.
**Enter o →** abre la ficha del rollo, con estado y avance en vivo, archivos
y acciones. **← o Esc** vuelve. El atajo **e** abre la misma ficha.

El listado muestra primero **Completos**, después **Procesando / En cola**,
luego **Incompletos / Con error** y al final **Sin descargar**. Dentro de cada
grupo se ordena alfabéticamente por rollo. El cursor conserva el rollo elegido
cuando un cambio de estado reordena el listado.

**Sin descargar** significa que no hay descarga iniciada ni trabajo activo.
**En cola** indica un trabajo solicitado que espera su turno; las etapas
numeradas indican un trabajo activo. Los estados **Falta…** indican trabajo por hacer, sin un
proceso activo. El filtro **Incompletos** reúne los rollos que aún no están listos.

El total de ZIP cuenta los archivos publicados en Archive para ese rollo.
Puede haber varios: por ejemplo, el rollo 284 tiene un ZIP del fichero
100501–101400 y otro llamado «101191 anexo». **0/2** significa que ninguno
de esos dos ZIP está descargado; **1/2**, que falta uno. Podés ver sus nombres
directamente en la ficha del rollo, junto con las rutas de ambos ZIP.

**Completo significa todas las páginas de todos los PDF de todos los ZIP del
rollo, guardadas en PNG a 200 DPI.** Es un formato común fijo. Las muestras,
otras resoluciones y conversiones personalizadas quedan disponibles como extras
y no se suman a la salida final.

El uso básico es **↑ / ↓ → Enter → Descargar y procesar**. No hay que marcar una
selección ni elegir etapas. La misma acción descarga los ZIP pendientes, extrae los
PDF y convierte todas sus páginas. La confirmación muestra el rollo, destino y
bytes pendientes. El estado pasa por En cola y las etapas numeradas, y termina en Completo
o Error. El listado se actualiza automáticamente durante el trabajo.

La ficha del rollo tiene dos bloques separados por una línea y un espacio:
**ARCHIVOS** y **ACCIONES**. Las flechas saltean los títulos y separadores.
Dentro de ARCHIVOS aparecen primero los elementos disponibles:

- **Carpeta de imágenes finales** y **muestras / otras conversiones**, identificadas por
  separado. Enter o → abre la carpeta y permite recorrer sus imágenes.
- **Cada PDF** y **cada ZIP** por separado, aunque el rollo tenga varios ZIP.
- La **ruta absoluta del elemento seleccionado**, a la vista debajo del listado.
  Si una ruta no cabe en una terminal pequeña, **p** la muestra completa.
- Los destinos todavía vacíos aparecen como **sin descargar**, **sin extraer**
  o **sin generar**, atenuados y deshabilitados. Las flechas los saltean; solo se
  pueden abrir archivos disponibles. Sin archivos, el cursor empieza en la acción principal.

Las acciones están debajo, en su propio bloque de la misma ficha:

- **Descargar y procesar / Continuar procesamiento / Reintentar procesamiento**:
  una sola acción, según el estado, que reutiliza originales y páginas válidas.
- **Eliminar datos locales**: tras confirmar, borra ZIP, PDF, conversiones y
  copias de recuperación de los archivos de ese rollo. Conserva catálogo y
  registros para volver a descargarlo. Si su trabajo está activo en esta máquina,
  lo detiene antes de borrar; comprueba la identidad del proceso para no detener
  otro trabajo. Un trabajo de otra máquina debe eliminarse desde esa máquina.
- **Herramientas y registros**: extracción solo de PDF, conversión personalizada,
  verificación de integridad, detalle técnico del trabajo y descarga de solo ZIP.

Después de completar, descargar o eliminar, la ficha permanece abierta y
actualiza su estado. **→ y Enter** también eligen opciones de los menús;
en los campos de texto, las flechas conservan su función de mover el cursor.
Al abrir un PDF o una imagen, se muestra el **visor dentro de la terminal**.
Las imágenes de una carpeta se pueden recorrer sin volver al listado; los PDF
se recorren por página. El visor muestra el número actual y el total.
Los ZIP muestran su contenido; abrirlos no extrae archivos.

### Visor de imágenes y PDF

- **←**: volver a la carpeta o ficha, como en el resto del explorador.
- **↑ / ↓** o **PgUp / PgDn**: imagen o página anterior/siguiente, también al ampliar.
- **Home / End**: primera/última. **g**: ir a un número concreto.
- **+ / −**: ampliar/reducir; **0**: ajustar a la ventana.
- **W / S**: desplazar arriba/abajo al ampliar; **A / D**: desplazar a los lados.
  También se conservan **h / j / k / l** como alternativas para desplazar.
- **p**: ruta completa; **?**: ayuda; **Esc / q**: volver a la carpeta o ficha.
- **v**: elegir el tipo de vista. **o**: abrir con `xdg-open` si la sesión tiene
  escritorio gráfico; abrir un archivo nunca lanza automáticamente ese programa.

El visor ofrece gráficos mediante los protocolos
[Kitty](https://sw.kovidgoyal.net/kitty/graphics-protocol/) e
[iTerm2](https://iterm2.com/documentation-images.html). Transmite los píxeles por
la terminal, por lo que también puede usarse por SSH; no necesita compartir una
ruta local con el emulador. La selección automática reconoce `TERM=xterm-kitty`,
`KITTY_WINDOW_ID`, `TERM_PROGRAM=iTerm.app` e `ITERM_SESSION_ID`. Si esos datos
no llegan por SSH, **v** permite elegir el protocolo manualmente.

En otras terminales se usa una **vista aproximada por caracteres**, con grises
cuando hay colores disponibles. Sirve para reconocer la página y recorrerla,
pero su resolución no alcanza para leer texto pequeño con la nitidez de una
imagen gráfica. Para esa lectura usar una terminal compatible con gráficos o
el visor de escritorio. Dentro de tmux/screen se elige inicialmente caracteres;
el modo gráfico manual depende del soporte y passthrough del multiplexor.
Si una vista gráfica no aparece, **v → vista aproximada por caracteres** permite
recuperar la visualización. Se puede fijar el modo para una sesión:

```bash
ARCHIVE_BERRUTTI_VIEWER=kitty archive-berrutti
# Otros valores: auto, iterm2, blocks.
```

La reducción al ajustar a la ventana usa filtrado Lanczos para evitar que
desaparezcan trazos finos o aparezcan bandas falsas. En terminales de 256 colores
se usan hasta 15 tonos, con blanco y negro completos. Esto mejora el muestreo y
el contraste, pero no añade resolución a los caracteres: cada celda sigue
representando dos muestras verticales. Una ventana con más filas/columnas o una
fuente de terminal más pequeña deja representar más detalle; el zoom lo dedica
a una región menor de la página.

Los PDF requieren Poppler (`pdfinfo` y `pdftoppm`, también usados para preparar
el dataset). Se renderiza **solo la página elegida**, con lado máximo de 2400
píxeles, en un directorio temporal privado que se elimina al cerrar el visor.
Zoom y desplazamiento reutilizan esa página; no convierten el PDF entero.
La visualización no modifica originales, manifiestos, imágenes finales ni el
estado de completitud del rollo. Un archivo ilegible muestra el error y permite
continuar a la imagen siguiente.

### Operaciones y seguimiento

Cada acción básica afecta al **rollo bajo el cursor**, incluidos todos sus ZIP.
Las selecciones guardadas por versiones anteriores no intervienen.

Los trabajos se procesan **de uno en uno por biblioteca, en segundo plano**.
Podés seguir explorando, completar más rollos o cerrar la interfaz; cada trabajo
conserva estado y log en `jobs/<id>/`. Con el perfil Ruche, cada uno es un batch
Slurm independiente del explorador, limitado por el tiempo del perfil worker.
El modo manual sin perfil conserva los procesos dentro de su asignación original.
Una interrupción se muestra como error recuperable y permite reintentar.

Durante la transferencia, la columna Estado muestra **1/3 Descarga 42.5%**;
durante la extracción, **2/3 Extrayendo PDF 0/1**; y durante la conversión,
**3/3 Creando imágenes**. También se distinguen conexión, verificación,
conteo de páginas y reintentos. El pie muestra una barra, porcentaje, tiempo
aproximado restante y **ZIP actual/total**. En la extracción, la barra sigue
los bytes descomprimidos aunque todavía no se haya terminado un PDF.
**Herramientas y registros → Ver registros y detalle técnico** amplía los
contadores y muestra también los bytes descargados
del rollo completo, incluidos los ZIP anteriores y parciales reanudables.

El tiempo estimado corresponde **a la etapa del ZIP actual**, no a las etapas
futuras ni a toda la cola. Se calcula con avance reciente; al comenzar aparece
**Estimando tiempo…** y se retira la estimación si deja de haber avance reciente.
Las páginas reutilizadas no cuentan como trabajo nuevo para calcular el ritmo.
Las conversiones ya iniciadas también muestran total y estimación al reabrir
la CLI: se leen sus inventarios y eventos existentes sin reiniciar el proceso.

Un 100% de bytes recibidos
no equivale a **Completo**: primero se verifica el ZIP y, para completar el
rollo, se extraen los PDF y se convierten todas las páginas.

**Herramientas y registros → Descargar solo ZIP** también trabaja en segundo plano;
vuelve a la ficha al confirmar. Podés cerrar y reabrir la CLI o abrir otra
terminal sobre la misma biblioteca: el progreso se lee de disco. Los trabajos
nuevos lanzados desde otra terminal aparecen automáticamente.

Cada descarga o trabajo de completar guarda en `jobs/<id>/`:

- `request.json`: rollo, archivos y acción solicitada.
- `state.json`: estado actual, último mensaje, archivo en transferencia,
  bytes recibidos, tamaño total e intento. También identifica la etapa, el ZIP
  actual y el inicio de esa etapa. Se reemplaza de forma atómica.
- `progress.jsonl`: historial de etapas y avance, aproximadamente cada segundo
  durante una transferencia, además de los cambios de etapa y los errores.
- `worker.log`: salida del proceso cuando se lanzó en segundo plano.
- `result.json`: resultado al terminar correctamente.

El detalle técnico muestra la ruta de esa carpeta. Los archivos `.zip.part` crecen dentro
de `rolls/.../zips/` hasta verificarse y publicarse como ZIP. El comando por lotes
`download` también registra el trabajo y puede monitorearse desde la CLI; su
salida de texto se muestra en la terminal que lo ejecuta.

Atajos opcionales: **/** busca por rollo/título; **c** completa; **e** abre la
ficha; **d** elimina; **f** filtra todos/completos/incompletos/con error/en proceso;
**r** relee los estados; **F2** abre biblioteca, destino y ayuda. **PgUp/PgDn**
pasa página y **Home/End** va al primer/último rollo. **Esc** vuelve o sale.
Dentro de la ficha, **p** muestra la ruta seleccionada y
**a** abre herramientas. Los menús se recorren con flechas y Enter.

La biblioteca y el catálogo quedan recordados. F2 → Cambiar carpeta abre otra
biblioteca; también se puede usar `browse --root /ruta/biblioteca`. Los campos
admiten flechas, Home/End, Backspace y Ctrl+U para vaciar. La primera entrada
obtiene solo el inventario; `browse --offline` impide actualizarlo por red.
Un snapshot local se importa con `refresh --metadata /ruta/metadata.json`.

La interfaz usa `curses` de Python para Linux; requiere al menos 70 columnas y
16 filas. Por SSH usar `ssh -t`. Sin terminal interactiva, usar comandos por lotes.

La salida final guarda un inventario con el total real de páginas obtenido de
cada PDF. Para mostrar Completo se comprueban perfil, cobertura de todas las páginas,
recibos y presencia/tamaño de cada imagen esperada. Completar verifica además
los hashes; la vista de estado evita recalcular todos los hashes en cada refresco.
Un reintento conserva evidencia dañada en `recovery/` o genera una nueva conversión,
y continúa con el trabajo válido. Solo se cuenta la salida final activa de cada
ZIP, sin sumar versiones anteriores.

En la biblioteca de ejemplo, los rollos **099 (1/1 ZIP)** y **284 (2/2 ZIP)**
están descargados y verificados. La ficha muestra el avance actual de extracción
y conversión. El rollo 321 conserva un ZIP de muestra y le falta otro.
Las tres imágenes de prueba de 284 y 321 aparecen como muestras en sus fichas;
no equivalen a la salida final de esos rollos.

## Uso por agentes

Los comandos sin destino explícito usan el mismo perfil predeterminado que el
explorador. Agregar `--local` para la biblioteca local o `--profile ruche` para
fijar Ruche. El campo JSON `location` identifica el host de las rutas devueltas.
Ver [uso remoto por agentes](REMOTE_RUCHE.md#uso-por-agentes), incluidos
`agent cancel ID` y `agent remove --rolls ... --yes`.

`archive-berrutti agent` es la entrada no interactiva. Comparte biblioteca,
descargas reanudables y trabajos con el explorador; no necesita terminal ni
respuestas a preguntas. No cambia la interfaz visual. También funciona con
`python -m archive_berrutti agent`. Usar `agent --help` y `agent files --help`
para descubrir comandos y filtros.

Los ejemplos siguientes usan la biblioteca recordada por el explorador. Todos
admiten `--root /ruta/biblioteca` **después del subcomando** para elegir otra.

```bash
# Qué hay localmente; estado final y próximo paso de cada rollo.
archive-berrutti agent list --state local
archive-berrutti agent list --state complete --limit 20 --offset 0
archive-berrutti agent list --state error

# Rutas de todos los ZIP/PDF y carpetas/manifiestos de imágenes de estos rollos.
archive-berrutti agent info --rolls 099,284

# Archivos presentes: rutas absolutas, tamaños y procedencia.
archive-berrutti agent files --rolls 099,284 --kind zip
archive-berrutti agent files --rolls 099 --kind pdf
archive-berrutti agent files --rolls 099 --kind image --limit 10
archive-berrutti agent files --rolls 284 --kind image --scope other

# Estimar sin escribir datos ni iniciar procesos.
archive-berrutti agent plan --rolls 099,284
archive-berrutti agent download --rolls 284 --dry-run

# Estas órdenes sí trabajan: elegir rollos antes de ejecutarlas.
archive-berrutti agent download --rolls 284 --background --max-download-gib 1
archive-berrutti agent complete --rolls 284 --background --max-download-gib 1

# Consultar el ID devuelto en data.results[].job.id, sin abrir la TUI.
archive-berrutti agent job ID_DEL_TRABAJO
```

Ejecutar **una acción a la vez por rollo**. `complete` ya incluye descargar;
no hace falta lanzar ambas órdenes. Con `--background`, la respuesta devuelve
IDs y rutas de registros inmediatamente; los procesos siguen después de salir
del comando. Sin esa opción, espera a terminar. La cola es la misma del
explorador: un rollo en ejecución por biblioteca. No se duplica un trabajo
activo. Para reanudar un trabajo fallido, repetir la acción después de consultar
su estado. Se conservan resultados válidos y evidencia previa. En el perfil
Ruche, `--background` presenta un batch independiente; en un batch manual
de `prepare.sbatch`, ejecutar la acción en primer plano.

`download` y `complete` exigen `--rolls` o `--all`; este último solicita **todo
el catálogo**. Incluyen todos los ZIP de cada rollo, también anexos. `99`
selecciona `099`, pero no `099r`. `complete` genera todas las páginas en PNG
a 200 DPI. `--max-download-gib` limita los **bytes ZIP pendientes estimados**,
descontando archivos descargados y parciales; no limita PDF ni imágenes, cuyo
espacio adicional se devuelve como `null` hasta conocerlo. Los presupuestos y
la selección se comprueban antes de iniciar trabajos. `--reserve-gib` (5 por
defecto), `--quota-free-gib` y `--attempts` se transmiten al trabajador.

Para una biblioteca nueva, inicializar el catálogo explícitamente:

```bash
archive-berrutti agent refresh --root /ruta/biblioteca
# Alternativa sin red:
archive-berrutti agent refresh --root /ruta/biblioteca --metadata /ruta/metadata.json
```

### Contrato de salida, versión 1

Salvo `--help`, **stdout contiene exactamente un documento JSON**, tanto en
éxito como en error. El progreso del modo en primer plano va a **stderr**;
en segundo plano queda en `worker.log`. No hay colores, menús ni prompts.

```json
{"schema_version":1,"ok":true,"command":"list","root":"/ruta/biblioteca","data":{"items":[],"total":0,"offset":0,"limit":50,"next_offset":null},"observed_at":"2026-10-08T12:00:00+00:00"}
```

- `list` devuelve estados por rollo: `complete`, `queued`, `processing`,
  `error`, `incomplete`, `not_downloaded`. `next_step` es `download`, `extract`,
  `render` o `null`. Incluye ZIP descargados/total, cantidad de PDF disponibles,
  imágenes finales disponibles/esperadas, errores y trabajo actual. El total
  esperado es `null` si todavía no se conocen todas las páginas. El filtro
  `local` excluye `not_downloaded`; `--search` busca rollo o título.
- `info` devuelve esos estados y `archives[]`, uno por ZIP: original, parcial,
  PDF, conversión final y otras conversiones. Todas las rutas son absolutas.
  `conversions[].recorded_images` cuenta registros del manifiesto; no es una
  garantía de presencia. `images.available` cuenta las imágenes finales
  válidas por presencia, tamaño y procedencia según la misma lógica de la TUI.
- `files` enumera archivos, no carpetas. Por defecto incluye los presentes y,
  para imágenes, solo la conversión final. `--scope other|all` incluye muestras
  u otras conversiones. `--availability missing|all` muestra originales o
  archivos registrados ausentes. `present`, `bytes`, `expected_bytes` y
  `size_matches` distinguen existencia de consistencia. `kind=zip_partial`
  identifica un ZIP en descarga; está incluido al pedir `--kind zip`.
  Las imágenes incluyen página, PDF de origen, formato, DPI, dimensiones y
  hashes registrados. Las imágenes se enumeran desde `pages.jsonl`; no se
  publican temporales todavía sin registrar. Los PDF ya publicados antes de
  terminar su manifiesto aparecen con `recorded=false`.
- `list` y `files` tienen paginación (`--limit 50` por defecto). `total` cuenta
  coincidencias y `next_offset=null` indica el final. El orden es estable por
  rollo y catálogo; los archivos de una conversión siguen su manifiesto.
  Las consultas reflejan observaciones de disco, no una instantánea atómica
  frente a trabajadores activos; repetir la consulta al terminar un trabajo.
- `job ID` devuelve estado, etapa, contadores, estimación temporal cuando está
  disponible y rutas de solicitud, estado, progreso, log y resultado. Algunas
  rutas pueden no existir aún; `worker.log` se crea en modo segundo plano.
  El ID sigue siendo consultable aunque se haya lanzado otro trabajo del rollo.
- Los códigos de salida son **0** (consulta/acción exitosa o trabajo aceptado),
  **1** (fallo operativo), **2** (argumentos inválidos), **130** (interrupción).
  Los errores tienen `ok=false` y `error.code/message`. Una acción con varios
  rollos conserva cada resultado en `data.results[]`, incluso si otro falla.
  **Consultar un job fallido devuelve 0** porque la consulta funcionó: comprobar
  `data.status.phase` y `data.status.error`. Una aceptación en segundo plano
  tampoco equivale a trabajo terminado; esperar `phase=done` y, para completar,
  comprobar `info` o `list` con `complete=true`.

`list`, `info`, `files`, `plan`, `job` y `--dry-run` usan la biblioteca seleccionada: no
consultan Archive ni inician descargas. Las consultas leen manifiestos y
comprueban existencia/tamaños; no recalculan hashes de todos los archivos.
Para verificar contenido usar `archive-berrutti verify --rolls 099`. Un listado
extenso de imágenes recorre sus manifiestos, pero la respuesta está paginada.

## Comandos por lotes

El equivalente del flujo básico, en primer plano (apto para `prepare.sbatch`), es:

```bash
archive-berrutti complete --root /ruta/biblioteca --rolls 284
archive-berrutti roll-status --root /ruta/biblioteca --rolls 284
```

`complete` exige `--rolls` o `--all` explícito; este último completa todo el
catálogo. Usa siempre PNG a 200 DPI y todas las páginas. `roll-status` devuelve
JSON con `complete`, `images`, `expected`, `zip_done`, `zip_total`, `pdfs`, errores
y trabajo actual. Las operaciones por etapas siguientes son opciones avanzadas:

Los comandos de lectura usan el último catálogo local y no consultan Archive.
Cambiar `ARCHIVE_DATA` por el destino deseado; elegir una carpeta nueva para una
biblioteca nueva. Cada operación conserva un plan con identificador nuevo.

```bash
ARCHIVE_DATA="$PWD/.private/archive_berrutti/mi-biblioteca"
archive-berrutti refresh --root "$ARCHIVE_DATA"
archive-berrutti list --root "$ARCHIVE_DATA" --search Fichero
archive-berrutti list --root "$ARCHIVE_DATA" --state downloaded
archive-berrutti list --root "$ARCHIVE_DATA" --state partial --json
archive-berrutti status --root "$ARCHIVE_DATA"
archive-berrutti info --root "$ARCHIVE_DATA" 9

archive-berrutti plan --root "$ARCHIVE_DATA" --rolls 99,100 --max-download-gib 10
archive-berrutti download --root "$ARCHIVE_DATA" --rolls 99,100 --max-download-gib 10
archive-berrutti extract --root "$ARCHIVE_DATA" --rolls 99,100
archive-berrutti render --root "$ARCHIVE_DATA" --rolls 99,100 --first-pages 5 --dpi 200
archive-berrutti verify --root "$ARCHIVE_DATA" --rolls 99,100
```

`list` admite `--offset` / `--limit` para paginación y estados `all`, `downloaded`,
`pending`, `images` y `partial`. El último incluye fallos y artefactos que requieren
revisión. `info --json`, `status` y `list --json` permiten inspección por otros scripts.

Para ejecutar toda la preparación de un lote, el equivalente es:

```bash
archive-berrutti prepare --root "$ARCHIVE_DATA" --rolls 99,100 \
  --first-pages 5 --format png --max-download-gib 10
```

La selección completa se estima con `plan --all`. `download --all` descarga
todos los ZIP; `complete --all` además extrae y convierte todas las páginas. No ejecutar esa descarga grande con la
cuota actual de 500 GiB de Ruche sin ampliar el espacio o reducir la selección.
`--all` abarca los ZIP del ítem del índice, no la colección mayor ni sus archivos
auxiliares de presentación (miniaturas, torrent y XML).

Selección: `--rolls` acepta sufijos exactos y normaliza ceros iniciales (`99`
coincide con `099`, no con `099r`). `--roll-range 99:110` incluye variantes con
sufijo. `--match` busca texto literal sin distinguir mayúsculas; los filtros se
intersectan. `--ids 1,3-5` selecciona por ID y no se combina con otros filtros.
`--limit` limita determinísticamente por orden de catálogo; no es una muestra
aleatoria. `--max-download-gib` rechaza selecciones mayores, sin recortarlas.

## Estructura de la biblioteca

```text
mi-biblioteca/
  library.json
  catalogs/<snapshot-id>.json
  active_catalog.json             # puntero mutable al último snapshot
  plans/<run-id>-<operación>.json
  operations/<roll-id>/current.json # último trabajo de cada rollo
  jobs/<run-id>/                    # solicitud, estado, log y resultado
  recovery/<run-id>/                # originales dañados conservados al reparar
  events.jsonl
  rolls/
    rollo-099/
      <título>-<id>-<hash>/
        selection.json
        final_target.json            # salida final activa del ZIP
        final_attempts/<run-id>.json  # perfil y total de páginas por PDF
        zips/<nombre-original>.zip
        receipts/<id>.json
        pdfs/<id>/
          0001-<nombre-original>.pdf
          manifest.json
        renders/<dpi>-<formato>-<config-id>/
          config.json
          render_plan.json
          images/<id-pdf-página>.jpg
          pages.jsonl
          complete.json
          events.jsonl
```

Se conservan los nombres originales de los ZIP y PDF; los miembros PDF se
aplanan y se numeran para impedir rutas peligrosas y colisiones. La carpeta
por título/ID/hash distingue varios archivos de un mismo rollo y nuevas versiones.
Añadir otro lote al mismo destino reutiliza originales completados: no crea
otra copia del mismo ZIP. Cada resolución, selección de páginas, versión del
código o de Poppler utiliza su propia conversión. No se reemplazan artefactos
completados ante una diferencia de hashes; cambiar el destino crea otra biblioteca.

## Reanudación, límites y procedencia

Las descargas conservan `.zip.part`, usan HTTP Range y verifican tamaño, MD5 y
SHA-1 antes de publicar un ZIP. Si el servidor ignora Range, reinician ese
parcial. Por defecto hay tres intentos por invocación (`--attempts`); el fallo
final queda visible y sale con código no cero. Un payload completo rechazado
se conserva para diagnóstico. Repetir el mismo comando reanuda o verifica las
etapas terminadas. Los locks impiden escritores simultáneos sobre la biblioteca.

La extracción conserva ZIP y PDF y verifica CRC/SHA-256. Un PDF parcial se
vuelve a extraer desde su inicio. Por defecto se rechazan PDF mayores de 8 GiB
(`--max-pdf-gib`). Seleccionar páginas reduce la conversión: **cada PDF elegido
requiere descargar primero su ZIP completo**.

Para imágenes: `--first-pages N` se adapta a PDF cortos; `--pages 1-5,10` exige
páginas existentes; `--all-pages` convierte todas. El valor inicial es las primeras
cinco páginas, JPEG a 200 DPI y calidad 95. `--format png` permite compresión sin
pérdida. Se conserva la página completa, con número de página desde 1. El máximo
**global por lote** es 1000 páginas (`--max-total-pages`), comprobado antes de
convertir. Cada página tiene un timeout de 180 s (`--timeout`).

`pages.jsonl` conserva procedencia, rollo, archivo original, miembro PDF, página,
dimensiones y hashes. `render_plan.json` registra las cantidades esperadas;
`complete.json` solo aparece al completar y verificar la selección. Un último
registro JSONL truncado se conserva como evidencia y se recupera al reanudar.
Errores de conversión se muestran y conservan las páginas ya terminadas.

Se comprueba espacio con 5 GiB de reserva (`--reserve-gib`). En WORKDIR de Ruche
se consulta la cuota normal, sus asignaciones pendientes y la cantidad de
archivos, usando el UID numérico en nodos de cómputo; no se utiliza el límite
duro como espacio garantizado. Las cuotas y
el espacio físico se comparten con otros procesos: estas comprobaciones no
reservan capacidad. `--quota-free-gib` impone un presupuesto adicional de descarga.
PDF, imágenes y temporales requieren espacio adicional al tamaño comprimido.
El procesamiento es secuencial y no necesita GPU.

Las imágenes quedan `unlabeled`; no se generan targets ni se crean splits.
Incorporarlas al entrenamiento exige transcripciones verificadas y auditar
identidad documental y duplicados entre rollos.

## Ruche

Para uso normal desde esta máquina basta `archive-berrutti`: el perfil configurado
abre el explorador local con índice por SSH y envía descargas independientes a
Slurm. Ver [configuración y operación remota](REMOTE_RUCHE.md). Lo que sigue es
la alternativa manual por etapas, útil para otros despliegues.

Usar la skill `ruche`. Transferir esta carpeta a un directorio de código nuevo en
WORKDIR y utilizar un Python >=3.11 ya preparado con Pillow. Cargar Poppler si
no están disponibles `pdfinfo` / `pdftoppm`. No instalar, escanear ni convertir
en el login. Los comandos rechazan los hosts de login conocidos.

```bash
export ARCHIVE_CODE_ROOT="$WORKDIR/archive-lote-001/code"
export ARCHIVE_PYTHON="/ruta/al/python-existente"
ARCHIVE_DATA="$WORKDIR/archive-lote-001/biblioteca"
mkdir -p "$WORKDIR/archive-lote-001/logs"

sbatch --test-only --output="$WORKDIR/archive-lote-001/logs/%x-%j.out" \
  "$ARCHIVE_CODE_ROOT/archive_berrutti/prepare.sbatch" \
  refresh --root "$ARCHIVE_DATA"
sbatch --output="$WORKDIR/archive-lote-001/logs/%x-%j.out" \
  "$ARCHIVE_CODE_ROOT/archive_berrutti/prepare.sbatch" \
  refresh --root "$ARCHIVE_DATA"
```

Pasar los demás comandos de la misma forma y esperar su éxito antes del paso
siguiente. La plantilla usa `cpu_short`, dos CPU, 8 GiB y 45 minutos. Ajustar
partición/tiempo para lotes medidos; las operaciones se reanudan tras walltime.
No solicitar SSD local en particiones CPU ni alterar jobs de entrenamiento.
Para explorar interactivamente en Ruche, usar una asignación CPU con `srun --pty`
y ejecutar `python -m archive_berrutti browse --root "$ARCHIVE_DATA"` allí.

## Verificación

`make verify` incluye selección, transferencias HTTP parciales/inconsistentes,
checksum, presupuesto, extracción segura, conversión de PDF sintéticos, biblioteca
incremental, presupuesto global de páginas, estados, selección interactiva y
cancelación sin descarga. La interfaz y las etapas se prueban además con una
muestra real pequeña.

`python -m archive_berrutti.smoke --root /ruta/nueva` inventaría todo y selecciona
los dos ZIP menores, con un presupuesto total de 2 MiB. Comprueba descarga,
reanudación, incorporación de otro rollo, extracción, dos imágenes JPEG, una PNG
y todos los hashes. Nunca inicia una descarga completa. En Ruche, ejecutarlo
únicamente bajo Slurm. Las evidencias se guardan dentro de esa biblioteca privada.

La prueba del flujo básico usa PDF sintéticos de varias páginas: completitud
por rollo con varios ZIP, exclusión de muestras, fallo durante conversión,
reanudación, imágenes faltantes, reparación, borrado acotado, cancelación del
trabajo propio, dos procesos reales en cola y navegación con teclado. También
se verifica el CLI por lotes y que un preview dañado no bloquee la salida final.

La muestra real previa se conserva: dos ZIP pequeños de los rollos 284 y 321,
780.051 bytes en total, con sus PDF y tres imágenes de prueba. La preparación
por etapas se ensayó localmente y en Slurm CPU de Ruche. La integración remota
agrega una prueba del flujo completo con el rollo 332 en la biblioteca de Ruche;
no se ejecutó una descarga completa del catálogo.

El comando previo `python -m experiments.archive_data.manage` sigue disponible
como compatibilidad del CLI inicial por etapas; para la biblioteca y el explorador,
usar `archive-berrutti` o `python -m archive_berrutti`.
