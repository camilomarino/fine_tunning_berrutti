# Berrutti v3 — 175 páginas

[Volver a los tres conjuntos](../README.md).

Es el conjunto usado para entrenar y evaluar el OCR de páginas del proyecto.
Contiene **175 imágenes de páginas con su transcripción humana completa** y
anotaciones que ubican las líneas dentro de cada página. Son 175 páginas, no
necesariamente 175 documentos históricos distintos.

La anotación VIA2 contiene **8.975 regiones rectangulares con texto humano**
repartidas entre las 175 páginas. Cada región guarda `x`, `y`, `width`, `height`
en píxeles de la imagen original y su transcripción en `region_attributes.text`.
El TXT de página es la referencia completa y conserva la secuencia textual
publicada. **El orden de las regiones del JSON no garantiza ese mismo orden de
lectura**; tampoco se asume que concatenarlas reconstruya siempre el TXT.
La alineación debe comprobarse antes de preparar un entrenamiento nuevo.

Procede de la versión Berrutti `v3.0` que ya usaba `berrutti_benchmark`. Se
conservaron imágenes, transcripciones, licencias y colaboradores originales.

## Dónde abrir los archivos

Hay una copia completa verificada **en esta máquina y otra en Ruche**:

| Ubicación | Carpeta de originales |
| --- | --- |
| Esta máquina, desde la raíz del repositorio | `data/berrutti_175/current/raw/` |
| Ruche | `$WORKDIR/berrutti-data/berrutti_175/current/raw/` |

Dentro de esa carpeta, abrir `images/` para ver las páginas,
`full_text_transcription/` para leer sus textos, o `bounding_boxes_lines/`
para consultar las anotaciones de líneas en formato VIA2. `licenses/` contiene
las condiciones originales de uso.

Para obtener las rutas absolutas, desde la raíz del repositorio en esta máquina,
con `.venv` activado:

```bash
python -m data path berrutti_175 --where local
python -m data path berrutti_175 --where ruche
```

La segunda ruta se abre en Ruche; el comando no descarga una copia local.

Para explorarlo visualmente, usar `berrutti-data browse berrutti_175`. Desde la
ficha de página se puede ver el texto, las cajas sobre la imagen y cada recorte
de línea con su transcripción. Para agentes, `berrutti-data agent annotations
berrutti_175 --id ID_PAGINA` entrega solo las regiones de esa página.
Ver el [uso de la CLI común](../CLI.md).

## Organización y uso en entrenamiento

```text
current/                              # enlace a una versión cerrada
  raw/
    manifest.json                     # manifiesto original
    images/
    full_text_transcription/
    bounding_boxes_lines/
    licenses/
    VERSION_AND_COLLABORATORS.txt
  records.jsonl                       # una fila por página, project_split explícito
  split-v1.json
  files.jsonl
  release.json
```

La copia local está físicamente en `data/berrutti_175/releases/`, excluida de
Git. `.private/dataset` conserva compatibilidad mediante un enlace a `current/raw`.
La fuente compartida con el benchmark sigue intacta. En Ruche hay otra copia
verificada dentro de `$WORKDIR/berrutti-data/berrutti_175/`; las corridas previas
conservan sus entradas originales.

La partición vigente contiene **125 páginas de entrenamiento, 25 de validación
y 25 de test**, agrupadas por rollo. Es idéntica a
[split-v1.json](split-v1.json).

Los scripts de entrenamiento siguen aceptando `--dataset` apuntando a **raw/**,
porque allí permanece el `manifest.json` compatible con `load_pages`.
Los crops, OCR auxiliar, manifiestos del currículo y cohortes emparejadas son
derivados versionados por corrida: no se trasladan dentro de esta fuente ni se
confunden con las líneas de LUISA. La identidad documental entre rollos y con
otros conjuntos sigue requiriendo auditoría.

## Identidad de la versión

- Versión de origen: `v3.0`.
- Identidad SHA-256 del contenido: `4af41cdd70771c0c8b0f3caa3739f534507d83a2cd3a49184f75f4af28161ab6`.
- `python -m data show berrutti_175` devuelve la ficha y las ubicaciones en JSON.

No se renombra ni transforma ninguna imagen o transcripción. Para interpretar
los índices, importar otra copia o verificar sus hashes, ver la
[guía técnica](../OPERACION.md).
