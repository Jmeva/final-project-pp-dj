# final-project-pp-dj

## Instrucciones para replicar

1. Clona el repositorio en la ubicación de tu preferencia dentro de tu ordenador con el comando `git clone` seguido de la liga SSH que proporciona GitHub: `git@github.com:Jmeva/final-project-pp-dj.git`.
2. En la raíz del repositorio clonado, crea el ambiente virtual con el comando: `python -m venv .venv`
3. En MacOS, actívalo con: `source .venv/bin/activate`
4. Cuando el ambiente esté activo, verás (.venv) al inicio de la línea en tu terminal. Una vez activo, instala las librerías con: `pip install -r requirements.txt`
5. Dentro de la carpeta `notebooks` abre el archivo `primer_modelo.ipynb` y corre por completo el notebook para reproducir el análisis.

## Nota para Dr. James Syme 
El notebook `primer_modelo.ipynb` es el documento base del proyecto y está organizado en cuatro secciones. Abre con la **carga de datos** desde `base_modelo_2019_2023.csv` y la construcción de los índices de grupo (estado, año, institución, tipo). Le sigue el **Modelo A**, que sirve como prior predictive check con una regresión logística jerárquica básica para verificar que las priors producen valores de θ razonables. El **Modelo B** reduce el análisis a las tres instituciones civiles para validar convergencia MCMC antes de incorporar la complejidad completa. Finalmente, el **Modelo C** es el modelo principal, el cual parte de una regresión logística bayesiana jerárquica con estructura civil/militar anidada mediante `ZeroSumNormal`. El modelo incluye diagnósticos de convergencia (`plot_trace`, `plot_ess`, `plot_autocorr`), posterior predictive check (`plot_ppc`) y forest plots por tipo e institución. 
