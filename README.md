# final-project-pp-dj

## Instrucciones para replicar

1. Clona el repositorio en la ubicación de tu preferencia dentro de tu ordenador con el comando `git clone` seguido de la liga SSH que proporciona GitHub: `git@github.com:Jmeva/final-project-pp-dj.git`.
2. En la raíz del repositorio clonado, crea el ambiente virtual con el comando: `python -m venv .venv`
3. En MacOS, actívalo con: `source .venv/bin/activate`
4. Cuando el ambiente esté activo, verás (.venv) al inicio de la línea en tu terminal. Una vez activo, instala las librerías con: `pip install -r requirements.txt`
5. Dentro de la carpeta `notebooks` abre el archivo `notebook_proyecto_final.ipynb`, verifica que el Kernel que identifica el notebook sea el del ambiente virtual y corre por completo el notebook para reproducir el análisis.
   1. En Visual Studio Code para seleccionar el Kernel se debe dar click en el botón que aparece en la parte superior derecha que dice `Select Kernel` y después buscar aquel que tenga el mismo nombre que el ambiente virtual (`.venv`) o en su defecto cualquier nombre que se le haya puesto al ambiente virtual. 

## Estructura del repositorio 

- **`data/`**: fuentes de datos crudas (archivos `.xlsx` de IMSS, IMSS Bienestar, SEDENA y SEMAR) y los datos ya procesados (`agg_folio_2018_2024.csv`, `base_modelo_2019_2023.csv`) que alimentan el modelo. También incluye `resumen_clases_pymc.md` con notas de referencia sobre PyMC.
- **`notebooks/`**: contiene distintas versiones de trabajo y exploración usadas durante el desarrollo del proyecto. **El notebook final es `notebook_proyecto_final.ipynb`**; el resto de los notebooks en esta carpeta son versiones intermedias o de exploración que se conservan como historial, pero no se seguirán actualizando.
- **`src/`**: módulos de Python con las funciones reutilizables del proyecto (carga de datos, procesamiento, definición de modelos y visualizaciones). Existen algunos archivos con nombres similares (p. ej. variantes con sufijo `_david` o `_jmv`) que corresponden a versiones de distintos integrantes del equipo; todos se conservan en el repositorio.
- **`imgs/`**: imágenes y diagramas usados en la documentación del proyecto (p. ej. `Data_access_pipeline.png`).
- **`requirements.txt`**: dependencias necesarias para reproducir el ambiente virtual del proyecto.
