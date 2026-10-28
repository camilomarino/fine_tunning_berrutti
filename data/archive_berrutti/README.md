# Archivo SID / Berrutti de Archive.org

[Volver a los tres conjuntos](../README.md).

Es la colección de rollos del [Archivo SID / Berrutti en Archive.org](https://archive.org/details/archivos-del-terror-de-uruguay_sid-berrutti_indice).
Cada rollo agrupa uno o varios ZIP que contienen PDF. La herramienta
[Archive Berrutti](../../archive_berrutti/README.md) los descarga, extrae los PDF
y genera las páginas finales como **PNG a 200 DPI**. Conserva ZIP, PDF e imágenes.
Este conjunto no incluye transcripciones humanas asociadas para entrenar OCR
supervisado.

Algunos PDF contienen una capa de texto extraíble: se comprobó en las primeras
tres páginas del rollo 099 el 9 de octubre de 2026. Su cobertura y calidad en
la colección completa no están auditadas, y ese texto automático no es una
transcripción humana de referencia. **Completar un rollo no ejecuta OCR**:
Poppler convierte sus páginas en imágenes. Los manifiestos de conversión
conservan PDF de origen, número de página, configuración y hashes; no se
crean cajas de líneas ni etiquetas nuevas a partir de esa conversión.

**La colección completa sigue en Archive.org; aquí se preparan rollos por partes.**
Estar en el catálogo no significa estar descargado. Un rollo está finalizado
cuando tiene todas sus imágenes finales; tener solo el ZIP no alcanza.

## Dónde están y cómo explorarlos

| Biblioteca | Ubicación |
| --- | --- |
| Ruche, predeterminada | `$WORKDIR/berrutti-data/archive/library-v1/` |
| Esta máquina, desde la raíz del repositorio | `data/archive_berrutti/local/` (enlace a la biblioteca bajo `.private/`) |

Son bibliotecas independientes; sus rollos descargados pueden ser distintos.
Desde la raíz del repositorio en esta máquina, con `.venv` activado:

```bash
berrutti-data browse archive_berrutti          # explorar los rollos de Ruche
berrutti-data browse archive_berrutti --local  # explorar los rollos locales
```

Elegir un rollo con ↑/↓, abrirlo con Enter o → y volver con ←. El listado muestra
el estado y el número de imágenes finales. La ficha del rollo permite acceder
a **imágenes, PDF y ZIP con sus rutas**, ver el avance, completar lo que falta,
reintentar o eliminar sus datos.

La entrada `berrutti-data` permite elegir este archivo junto con Berrutti 175
y LUISA. Para agentes, `berrutti-data agent … archive_berrutti` conserva las
opciones del motor Archive. Ver la [guía unificada](../CLI.md).
`archive-berrutti` y `python -m archive_berrutti` siguen disponibles.

## Encontrar archivos desde un script o agente

```bash
python -m data show archive_berrutti
python -m data path archive_berrutti --where ruche
python -m archive_berrutti agent list --state complete
python -m archive_berrutti agent info --rolls 099
python -m archive_berrutti agent files --rolls 099 --kind zip
python -m archive_berrutti agent files --rolls 099 --kind pdf
python -m archive_berrutti agent files --rolls 099 --kind image --limit 10
```

`099` es un ejemplo de selección. Los comandos `agent` devuelven JSON; la lista
de archivos permite obtener las rutas exactas sin recorrer carpetas a mano.
Agregar `--local` antes de `agent` para consultar la biblioteca local.
El explorador y las consultas de lectura de `agent` no reservan CPU en Ruche;
el procesamiento se ejecuta bajo Slurm. Ver [operación remota](../../archive_berrutti/REMOTE_RUCHE.md).

## Organización y relación con los otros conjuntos

Dentro de `rolls/`, cada rollo conserva subcarpetas `zips/`, `pdfs/` y
`renders/…/images/`, separadas por archivo y configuración de conversión. La
[estructura completa](../../archive_berrutti/README.md#estructura-de-la-biblioteca)
explica esas rutas. El perfil y el `remote.json` privado registran la raíz real.
Se conserva su ruta física porque las solicitudes de trabajos contienen rutas
absolutas.

Los 175 ejemplos etiquetados y las líneas LUISA pueden compartir material de
origen con este archivo. Conservar los identificadores de rollo y auditar la
identidad documental antes de generar targets o mezclar ejemplos con validación/test.
