# Preparación de datos

Código CPU que produce corpus sellados, sin entrenar modelos ni importar el
runner. Las salidas se crean en rutas privadas nuevas.

- `contracts.py`: comprobación del cohort Berrutti, fuentes, split y caché de
  líneas, con los mismos criterios del experimento cerrado.
- `luisa.py`: política explícita de inclusión de LUISA, solapamientos, curriculum
  mixto y cobertura completa de las composiciones.
- `synthetic_pages.py`: crops originales, geometría, referencias y páginas
  sintéticas reproducibles.

```bash
python -m data.preparation.luisa --help
```

Los hashes y recibos inmutables están en `data/integrity.py`, la carga Berrutti
en `data/berrutti.py` y las anotaciones en `data/annotations.py`. Las recetas
consumen estos módulos; la preparación no depende de `finetune` ni `reporting`.
