# Fine-tuning Berrutti con Qwen3.8

Proyecto independiente para entrenar y evaluar Qwen3.8-27B como OCR de páginas
del conjunto Berrutti v3. El código proviene del experimento privado desarrollado
en el worktree de `berrutti_benchmark`; las corridas ya ejecutadas en Ruche son
evidencia inmutable y no se mueven ni se sobrescriben.

La receta actual usa BF16, dos A100 de 40 GB, LoRA r16/alpha32 en el
decodificador y encoder visual congelado. El split provisional contiene 125
páginas de entrenamiento, 25 de validación y 25 de test, agrupadas por rollo.
El CER/WER micro aplica la misma política `ascii-text-v1` del benchmark:
AnyAscii seguido de colapso de espacios. La implementación local está en
`experiments/berrutti_qwen38/scoring.py`; se conserva para que el proyecto no
importe el paquete del benchmark. Ver
`experiments/berrutti_qwen38/README.md` para la receta y las guardas.

## Estado heredado

La corrida de páginas completas terminó tres épocas. Frente al modelo base,
el CER de validación bajó de 2,746 % a 2,184 % y el de test de 5,027 % a
3,437 %, con 25 páginas completas en cada conjunto final. El CER de test de
la época 2 quedó provisional porque una página no terminó después de dos
intentos. El ensayo pequeño del currículo falló antes de cargar el modelo:
PyTorch no vio ninguna GPU en esa asignación. La corrida grande del currículo
permanece pendiente por dependencia. Ver `docs/HANDOFF.md` para decisiones y
próximos pasos.

El dataset licenciado, los crops, las mediciones previas y una copia privada
de la conversación (JSONL y Markdown) están bajo `.private/`, excluido de Git. `.private/dataset` es
un enlace a la copia local existente; los modelos, entornos y checkpoints
siguen en Ruche y no se descargan aquí.

## Entorno local

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,monitor]'
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Para usar `make verify`, activar antes `.venv`. El entorno GPU de Ruche es
separado y fija PyTorch 2.9.1+cu128, Transformers 5.19.0, PEFT 0.21.2 y
flash-linear-attention 0.5.2, más el kernel `causal-conv1d` ensayado. La copia
privada de la evaluación de viabilidad contiene los probes y el inventario de
paquetes. Transferir el código a una **ruta de corrida nueva** en WORKDIR y
validar primero en una asignación Slurm; nunca ejecutar cómputo en el login.

El monitor de dos horas ejecuta los scripts de este proyecto y sigue enviando
informes al chat original mientras se prepara el cambio de conversación. Los
temporizadores siguen activos; su configuración y estado permanecen en el
usuario, con copia de respaldo en `.private/monitor/`.
