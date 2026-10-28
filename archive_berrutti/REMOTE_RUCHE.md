# Biblioteca remota y perfiles de servidor

La CLI admite una biblioteca local y perfiles **SSH + Slurm**. En esta máquina
el perfil predeterminado es **`ruche`** y guarda los datos en:

```text
$WORKDIR/berrutti-data/archive/library-v1
```

`WORKDIR` se resuelve al configurar el perfil; la ruta absoluta, cuenta y Python
reales se consultan con `archive-berrutti profile show`. La configuración privada
vive en `.private/archive_berrutti/profiles.json`, fuera de Git. Un clon nuevo
sin esa configuración empieza en modo local, no hereda credenciales ni rutas de
otra persona. También se puede elegir su archivo con `ARCHIVE_BERRUTTI_CONFIG`.

## Uso diario

Desde la raíz del repositorio, con el entorno del proyecto activo:

```bash
source .venv/bin/activate
archive-berrutti                       # destino predeterminado: Ruche
archive-berrutti --profile ruche        # Ruche explícitamente
archive-berrutti --local                # biblioteca local recordada anteriormente
archive-berrutti --local browse --root /ruta/local/biblioteca
archive-berrutti profile show           # JSON: configuración y destinos
```

También funciona `python -m archive_berrutti`. `--profile` y `--local` se aceptan
antes o después del subcomando. Un `--root` explícito **sin `--profile` selecciona
local**, para conservar el significado de scripts existentes. Para otra carpeta
remota usar `--profile ruche browse --root /ruta/absoluta/dentro/de/WORKDIR`.
Los argumentos de archivos, como `--metadata`, se interpretan en el servidor
seleccionado; no se suben archivos locales implícitamente.

Para cambiar el destino predeterminado de forma persistente:

```bash
archive-berrutti profile use local
archive-berrutti profile use ruche
```

Un fallo de conexión remota se informa como error. **Nunca se cambia a local
silenciosamente ni se empieza a descargar en esta máquina.** La biblioteca local
previa se conserva, sin trasladarla ni mezclarla con la nueva biblioteca remota.

El listado, las flechas, Enter, las fichas de rollo y los visores de PDF/imágenes
son los mismos. La línea de carpeta identifica servidor y ruta. «Cambiar carpeta»
dentro de una sesión remota cambia a otra carpeta de WORKDIR del mismo servidor;
para cambiar entre servidores/local, cerrar y abrir con el perfil elegido.
Las rutas de las fichas remotas pertenecen a Ruche.

## Cómo se ejecuta

El **explorador se ejecuta en esta máquina y no reserva CPU de Ruche**. Lee por
SSH un índice publicado en `browse/`, sin montar GPFS ni recorrer el dataset en
el login. Las consultas `agent list/info/files/plan/job` y `--dry-run` usan el
mismo índice. Los trabajadores actualizan el rollo que procesan aproximadamente
cada cinco segundos y al terminar. El cliente consulta un pequeño indicador de
revisión; solo vuelve a traer el índice cuando cambia, y los listados de archivos
de los rollos que se abren o que intervienen en una página de resultados.

| Operación | Recursos iniciales del perfil Ruche |
| --- | --- |
| Explorador y consultas `agent` de lectura | Sin asignación Slurm |
| Publicar índice, actualizar inventario o enviar acciones | `cpu_med`, 1 CPU, 2 GiB, hasta 10 minutos |
| Descarga/conversión | `cpu_med`, 2 CPU, 8 GiB, hasta 4 horas por trabajo |
| Visor remoto opcional para documentos grandes | `cpu_med`, 1 CPU, 2 GiB, mientras esté abierto, hasta 4 horas |

Los tiempos son límites, no reservas de inicio inmediato. Descargar, extraer,
convertir, verificar o borrar datos sigue requiriendo cómputo, al igual que los
comandos antiguos de consulta fuera de `agent`. Para consultas automáticas usar
la API `agent`. El perfil usa `cpu_med` para que el envío y su trabajador no
compitan por el límite de un trabajo de `cpu_short`. Las particiones y cuenta
deben revalidarse al configurar otro servidor.

La acción única «Descargar y procesar», «Continuar procesamiento» o «Reintentar
procesamiento», según el estado, envía un **batch independiente**. Se puede cerrar
el explorador o la conexión y el trabajo continúa. `agent ... --background`
hace lo mismo. Los comandos en primer plano permanecen ligados a su invocación;
para tareas largas usar el modo de segundo plano.

Las solicitudes se serializan por biblioteca con bloqueo y dependencias
`afterany`: un fallo no deja las siguientes esperando éxito. Slurm conserva la
cola sin asignar CPU a trabajadores que esperan su predecesor. Los bloqueos de
datos existentes siguen protegiendo las operaciones. La cancelación valida
usuario e identidad de la solicitud y actúa solo sobre el ID Slurm asociado.

El estado combina recibos locales en GPFS con `squeue` y `sacct`. Un job pendiente
es «En cola», aunque no tenga PID. TIMEOUT, cancelación y fallos quedan visibles;
reintentar conserva lo ya verificado. La consulta al scheduler se limita a una
cada 15 segundos por proceso; los archivos de avance se leen entre consultas.
Si se pierde la respuesta de `sbatch`, se conserva la solicitud y se intenta
recuperar su ID por su nombre único, sin reenviarla automáticamente. Si Slurm no
permite confirmar identidad o terminación, se conserva la información y no se
borra. El detalle muestra la advertencia para volver a consultar.

Al elegir una imagen o PDF pequeño se trae **solo ese archivo**, para verlo
localmente. La caché privada admite como máximo 64 MiB por archivo y 128 MiB en
total, y se limpia al cerrar el explorador. Al recorrer imágenes se transfiere
solo la elegida. Para un documento mayor, o si falta Poppler local para PDF,
se ofrece abrir el visor bajo Slurm: esa vista sí requiere CPU mientras esté
abierta. Los modos por caracteres, Kitty e iTerm2 se conservan. No se expone un
servidor web ni se copia el corpus al cliente.

La cabecera muestra la fecha del índice. Si se pierde la conexión durante la
exploración, conserva el último estado y muestra una advertencia. Los cambios
hechos fuera de esta CLI, o por versiones anteriores de sus trabajadores,
requieren republicar el índice. Si un trabajo acaba sin publicar su estado final,
se informa para revisarlo; el código de salida de Slurm por sí solo no prueba
que todas las imágenes estén completas.

Para crear el índice de una biblioteca anterior o reconstruirlo:

```bash
archive-berrutti agent index                 # requiere CPU; no descarga ZIP
archive-berrutti agent index --rolls 332     # actualizar un rollo existente
```

La primera publicación recorre todos los rollos, aunque se indique una selección.
La biblioteca predeterminada de esta máquina ya tiene el índice inicial.

## Uso por agentes

El contrato sigue siendo JSON versión 1, un documento por comando en stdout,
con diagnósticos de SSH/Slurm en stderr. `location` identifica perfil, host,
raíz y scheduler; las rutas devueltas son absolutas **en ese host**. Los
comandos de consulta paginan los archivos igual que en modo local. Las respuestas
del índice añaden `snapshot_at`, `warning` y `location.access=published_index`:
`observed_at` es la fecha de consulta, no una comprobación nueva de cada archivo.

```bash
archive-berrutti agent list --limit 20
archive-berrutti agent list --state complete
archive-berrutti agent info --rolls 332
archive-berrutti agent files --rolls 332 --kind pdf
archive-berrutti agent files --rolls 332 --kind image --limit 10
archive-berrutti agent plan --rolls 332
archive-berrutti agent complete --rolls 332 --background --max-download-gib 0.002
archive-berrutti agent job ID_DE_SOLICITUD
archive-berrutti agent cancel ID_DE_SOLICITUD
archive-berrutti agent remove --rolls 332 --yes
```

Esos comandos usan el perfil predeterminado; agregar `--profile ruche` para
fijarlo o `--local` para el disco local. `ID_DE_SOLICITUD` es el ID devuelto en
`data.results[].job.id`, **no** el número Slurm. Este último está en
`job.status.slurm_job_id`. `remove` exige `--yes` y no detiene un trabajo activo
salvo que se agregue `--stop-active`. Consultar `info` antes de borrar.

No generar lanzamientos duplicados al ver «En cola» o una conexión fallida:
consultar el trabajo existente. `profile show` es local y no crea asignaciones.
Las consultas de lectura de `agent` no crean asignaciones. Las acciones y
`agent index` sí pueden esperar CPU.

## Configurar otro servidor o actualizar el código

Requisitos: alias SSH autenticado con verificación de host, WORKDIR accesible
por login y cómputo, Slurm (`srun`, `sbatch`, `squeue`, `sacct`, `scancel`), rsync,
Python >=3.11 con Pillow y curses, y Poppler (`pdfinfo`, `pdftoppm`) en cómputo.
Preparar dependencias dentro de una asignación; no instalar en el login ni
modificar entornos de entrenamientos activos.

Primero consultar identidad, asociaciones, partición, cuota y WORKDIR por SSH.
Registrar las rutas resueltas y la cuenta de ese usuario, por ejemplo:

```bash
archive-berrutti profile configure servidor \
  --host alias-ssh --workdir /ruta/workdir \
  --root /ruta/workdir/berrutti-data/archive/library-v1 \
  --python /ruta/al/entorno/bin/python --account proyecto \
  --session-partition cpu_med --session-time 04:00:00 \
  --worker-partition cpu_med --worker-time 04:00:00
archive-berrutti profile deploy servidor
archive-berrutti --profile servidor agent refresh
archive-berrutti --profile servidor agent list --limit 2
archive-berrutti profile use servidor
```

Los valores son ilustrativos. Hacer `sbatch --test-only` con los recursos de ese
perfil antes del primer lanzamiento; después ensayar con una selección pequeña.
No configurar un servidor de ClusterUY para este proyecto: su entrenamiento y
preparación remota se realizan en Ruche según `AGENTS.md`.

`profile deploy` copia **solo el paquete** a una versión nueva bajo
`$WORKDIR/berrutti-data/code/<release-id>/`, con perfil y hashes de los archivos.
No copia `.private/`, datasets, credenciales, modelos ni entornos. No construye
software en el login. La configuración local se actualiza al terminar la copia;
los trabajos y sesiones existentes conservan su versión de código. Después de
editar código, ejecutar pruebas, desplegar una versión nueva y abrir una sesión
nueva. No editar versiones remotas utilizadas por trabajos activos.
Al actualizar desde una versión sin índice publicado, ejecutar una vez
`archive-berrutti --profile servidor agent index` antes de abrir el explorador.
Una sesión abierta con código anterior conserva su comportamiento; cerrarla y
volver a abrirla para usar el explorador local.

## Datos, cuota y convivencia

```text
$WORKDIR/berrutti-data/
  code/<release-id>/                 # código y perfil de esa versión
  archive/library-v1/
    library.json
    catalogs/                       # inventarios
    rolls/                          # ZIP, PDF y conversiones
    jobs/<request-id>/               # solicitud, estado, log y recibo Slurm
    operations/                     # punteros, bloqueos y cola de biblioteca
    browse/index.json               # estados y recuentos publicados
    browse/revision.json            # indicador pequeño para actualización
    browse/rolls/                   # rutas y metadatos de archivos por rollo
```

Los grandes archivos viajan directamente **Archive.org → HTTPS → nodo CPU →
WORKDIR**. El cliente transporta código, JSON y archivos elegidos para visualizar
por SSH. En la conexión
comprobada de esta máquina, el alias `ruche` usa TCP **22**, sin ProxyJump ni
ProxyCommand. Rsync sobre SSH utiliza el mismo acceso, no un daemon en el puerto
873. Archive usa HTTPS saliente; no hacen falta puertos entrantes adicionales.

Ruche no hace respaldo automático de HOME/WORKDIR. La cuota normal consultada
es de 500 GiB y se comparte con entrenamiento. El inventario observado el
8 de octubre de 2026 suma 1.573 rollos y 1.602 ZIP, 357,75 GiB comprimidos; en
esa consulta ya superaba el margen restante. PDF, PNG, temporales y checkpoints
necesitan espacio adicional y también cuentan para la cuota de archivos.

No lanzar todo el catálogo con la cuota actual. Medir expansión y procesar por
lotes con margen para entrenamientos. `--max-download-gib` limita ZIP pendientes,
no el tamaño final; la comprobación remota de cuota/archivos y la reserva siguen
vigentes, pero no reservan capacidad frente a otros trabajos.

El conjunto etiquetado de 175 páginas y los resultados heredados permanecen en
sus ubicaciones/versiones existentes. Esta biblioteca contiene material
`unlabeled`; no altera etiquetas, splits, checkpoints ni jobs de entrenamiento.
La completitud exige ZIP, PDF y todas las imágenes PNG a 200 DPI. Borrar etapas
intermedias no es una política de ahorro implementada.

## Verificación y evidencia

`make verify` cubre modo local, rutas, JSON, selección de perfiles, invocación
Slurm, cola sin PID, dependencias, respuesta de envío perdida, TIMEOUT y
cancelación limitada al trabajo propio, lectura remota sin asignaciones,
publicación incremental, caché acotada y navegación que saltea archivos ausentes.
La configuración debe probarse además
en el servidor: los tests simulados no validan sus cuotas, red o runtime.

La evaluación anterior y el despliegue tienen registros privados bajo
`.private/archive_berrutti/remote-assessment-*/`, `remote-setup-*/` y
`remote-browser-*/`. Las pruebas
reales usan identificadores nuevos; no se reemplaza la evidencia de pruebas
previas. La muestra del rollo 332 sirve para comprobar el recorrido completo y
la visualización sin descargar todo el catálogo.

Referencias: [conexión y almacenamiento de Ruche](https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/03_connection_and_file_transfer/),
[Slurm en Ruche](https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/),
[sbatch](https://slurm.schedmd.com/sbatch.html),
[catálogo Archive](https://archive.org/metadata/archivos-del-terror-de-uruguay_sid-berrutti_indice).
Aplicar la skill `ruche` y las instrucciones de `AGENTS.md` al operar el servidor.
