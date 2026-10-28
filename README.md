# Fine-tuning y experimentos Berrutti

## Organización del código

| Área | Ubicación | Responsabilidad |
| --- | --- | --- |
| Datos | `data/` y `archive_berrutti/` | Catálogo, acceso, importación, imágenes, anotaciones y preparación CPU |
| Entrenamiento | `finetune/` | Preparación Slurm común y recetas en `finetune/recipes/` |
| Evaluación | `evaluation/` | Normalización, CER/WER y selección de checkpoint |
| Reportes y publicación | `reporting/` | Auditoría de resultados cerrados, estadísticas y viewers |
| Informes finales | `reports/` | Conclusiones y figuras versionadas, sin OCR ni datos originales |

La [guía de organización](docs/REPOSITORY_STRUCTURE.md) describe las dependencias,
la compatibilidad con comandos anteriores y cómo incorporar otra receta.
Los datasets reales, checkpoints, predicciones y perfiles de máquina quedan
fuera de Git. El código de datos no importa entrenadores; los entrenadores no
importan viewers ni herramientas de publicación.

## Entrenamientos

La entrada común [finetune](finetune/README.md) prepara trabajos Slurm para
Ruche y ClusterUY, separando el perfil de cómputo de la receta. Para modelos
generales de texto o imagen más texto reutiliza la CLI de TRL; admite también
recetas propias. No duplica entrenadores por familia. El soporte de preparación
está probado en CPU; cada nuevo modelo/entorno requiere ensayo GPU en el destino.

Comparación de tres modelos OCR, usando únicamente Berrutti. Cada modelo parte
independientemente del mismo Qwen y de la misma inicialización LoRA nueva.

| Experimento | Imágenes de entrenamiento en **cada** época | Épocas |
| --- | --- | --- |
| Sólo páginas | Las 125 páginas completas | 3 |
| Sólo líneas | Los 5.212 recortes de líneas de esas 125 páginas | 3 |
| Combinado | Las 125 páginas completas y sus 5.212 recortes | 3 |

Cada época recorre todos los ejemplos. El combinado intercala una página
completa y todas sus líneas, con peso 50/50 por página de origen. Los tres usan
las mismas 25 páginas completas de validación y 25 de test. La validación elige
el checkpoint; el test sólo describe su resultado.
La línea base se mide una sola vez. Tras cada época se evalúa sólo validación;
el test se ejecuta al final para el checkpoint elegido de cada modelo.
Son 350 evaluaciones de página entre los tres, frente a 500 con test por época
(reintentos aparte). El perfil común pasa por un ensayo representativo bajo Slurm.

El [protocolo](docs/THREE_ARMS_125_PROTOCOL.md) explica las épocas, los steps,
las discrepancias entre anotaciones y las condiciones de ejecución.
**Estado al 22 de octubre: los cuatro entrenamientos están cerrados y auditados.**
La [auditoría final](reports/berrutti/FINETUNING_AUDIT_20261022.md) verifica 208 checkpoints y
recalcula 825 casos únicos de evaluación de página. El combinado Berrutti obtiene
el mejor CER de validación (1,930 %); páginas es la alternativa de menor coste
observado. El test de líneas conserva una página incompleta. La extensión LUISA
mejora su propia baseline, con las limitaciones exploratorias indicadas abajo.
El seguimiento es sólo manual; no hay llamadas automáticas al modelo.

## Datos y biblioteca

Los datasets se consultan mediante [data/README.md](data/README.md) y
`python -m data list/show/path`. Imágenes, textos, crops, modelos, entornos,
checkpoints y ubicaciones privadas quedan fuera de Git.

**Explorador común:** activar `.venv` y ejecutar `berrutti-data` o `python -m data`.
Permite elegir entre los tres conjuntos, ver imágenes y textos, explorar las
líneas con cajas de Berrutti 175 y gestionar rollos de Archive. Para agentes:
`berrutti-data agent list`. Ver la [guía de la CLI unificada](data/CLI.md).

- **Berrutti:** 175 páginas con etiquetas humanas, split por rollo 125/25/25.
- **LUISA:** conjunto externo de imágenes de líneas. Está fuera de los tres
  experimentos actuales; sus particiones originales se conservan.
- **Archive Berrutti:** biblioteca de rollos sin etiquetas. Sus herramientas
  independientes están en [archive_berrutti/README.md](archive_berrutti/README.md).

El [experimento cuatro](docs/EXPERIMENT_4_LUISA_SYNTHETIC.md), autorizado por
separado, incorpora todas las líneas LUISA y páginas sintéticas, además de
Berrutti train completo. Completó su ensayo, baseline y 64 updates de entrenamiento. Sus resultados son exploratorios: incluye solapamientos conocidos
con test Berrutti y no certifica mejoras independientes.

## Web de resultados

El [OCR Diff Viewer de base y fine-tuneos](https://iie.fing.edu.uy/~cmarino/ocr/qwen_berrutti_comparison/)
está publicado y verificado. Usa directamente la interfaz v2, con base y ajustes
sobre las 50 páginas de validación/test. Los nombres indican los datos de
entrenamiento y las épocas. Muestra un único Qwen sin entrenamiento, con
generación determinista, y los cuatro ajustes. Conserva líneas provisional y LUISA
exploratorio con base propia. Sustituye la portada inicial descartada por el
usuario; el v2 de referencia permanece intacto. Ver la
[verificación de la publicación](docs/FINETUNING_PUBLICATION.md).

La [interfaz de resultados](docs/RESULTS_VIEWER.md) permite recorrer train,
validación y test de los tres modelos, comparar imágenes, referencias humanas
y transcripciones, y consultar épocas y baseline. Incluye los recortes de
entrenamiento y puede incorporar todas las líneas LUISA y composiciones del
experimento cuatro. Los resultados incompletos conservan su estado provisional.

El exportador crea un bundle estático desde evidencia cerrada, sin ejecutar
OCR, y prepara una publicación independiente en Ampere mediante el CLI
transaccional del benchmark. La guía explica la vista previa, las verificaciones
y el dry-run obligatorio antes de activar una release aprobada.

## Entorno y verificación

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,monitor]'
. .venv/bin/activate
make verify
```

La comparación Qwen–Berrutti enviada se ejecuta bajo Slurm en Ruche. Las nuevas
recetas pueden prepararse para ambos clusters. No se entrena en esta máquina
ni en nodos de login. Antes de un completo se requiere asignación GPU,
smoke y ensayo representativo. Las herramientas compartidas y el comando para
sellar las entradas están en [la guía de la receta](experiments/berrutti_qwen38/README.md).
El [handoff](docs/HANDOFF.md) mantiene únicamente el estado operativo vigente.
