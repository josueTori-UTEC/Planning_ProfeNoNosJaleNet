# Laboratorio 3: Planning (Qwen3-8B)

## Integrantes
**grupo ProfeNoNosJaleNet:**
- Rodrigo Meza  
- Josue Toribio

## Nota sobre tiempo de ejecución (Colab)
El proyecto cumple el requisito de **tiempo máximo < 2 minutos** para la ejecución principal de evaluación/generación de resultados (`submit.py`), **si el modelo Qwen3-8B ya se encuentra descargado en caché**, tal como ocurre normalmente al correr varias veces en el mismo entorno o cuando se reutiliza el caché.

- **Primera ejecución en un runtime limpio:** puede demorar más debido a la **descarga inicial del modelo** (≈ 16.4GB) desde Hugging Face.
- **Ejecuciones posteriores (inferencia / pruebas):** con el modelo ya descargado, el tiempo de ejecución de `submit.py` y `dev_test.py` se mantiene **por debajo de 2 minutos**, cumpliendo lo estipulado.

Recomendación: para evitar re-descargas, se puede reutilizar el caché del modelo entre sesiones (por ejemplo, usando almacenamiento persistente como Google Drive en Colab).
