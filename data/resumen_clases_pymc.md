# Resumen de PyMC — Clases 3 a 7

Contexto del curso: *Programación Probabilística* (PyMC + ArviZ).  
Libro de referencia principal: **Martin, Kumar & Lao — "Bayesian Modeling and Computation in Python"** (1ª ed.).

---

## Clase 3 — Redes Neuronales Bayesianas

**Tema:** Clasificación binaria con incertidumbre, Inferencia Variacional (ADVI).

### Patrón general PyMC

```python
with pymc.Model(coords=dict_coords) as model:
    # pymc.Data para inputs con dimensiones nombradas
    x_data = pymc.Data("X_data", x_train, dims=("obs_id", "feature_index"))
    y_data = pymc.Data("Y_data", y_train, dims="obs_id")

    # Minibatch para datasets grandes
    ann_input, ann_output = pymc.Minibatch(x_data, y_data, batch_size=50)

    # Priors sobre pesos (distribuciones normales)
    weights_in_1 = pymc.Normal("weights_into_1", 0, sigma=1,
                                initval=init_1,
                                dims=("feature_index", "hidden_layer_1"))

    # Activaciones usando pymc.math
    act_1 = pymc.math.tanh(pymc.math.dot(ann_input, weights_in_1))
    act_out = pymc.math.sigmoid(pymc.math.dot(act_2, weights_2_out))

    # Likelihood binaria
    out = pymc.Bernoulli("out", act_out,
                          observed=ann_output,
                          total_size=x_train.shape[0])  # obligatorio con Minibatch

# Inferencia Variacional (más rápida que MCMC para redes)
with model:
    approx = pymc.fit(n=100_000)

trace = approx.sample(draws=5000)

# Posterior predictive en modelo nuevo (sin Minibatch)
with pymc.Model(coords=coords):
    weights_in_1 = pymc.Flat("weights_into_1", dims=(...))
    ...
    ppc = pymc.sample_posterior_predictive(trace)
```

### Conceptos clave
| Concepto | Herramienta |
|---|---|
| Inferencia variacional | `pymc.fit(n=...)` → `approx.sample()` |
| Minibatch | `pymc.Minibatch(x, y, batch_size=...)` |
| Activaciones | `pymc.math.tanh`, `pymc.math.sigmoid`, `pymc.math.dot` |
| Prior plano (predicción) | `pymc.Flat(...)` |
| Visualizar convergencia VI | `plt.plot(approx.hist)` → curva ELBO |

### Archivo auxiliar (`pymc_nn.py`)
- Encapsula la construcción del modelo en `construct_nn()` y la predicción en `sample_posterior_predictive()`.
- Define constantes privadas para nombres de variables/dimensiones (`_VARNAME_*`, `_DIM_*`).
- Usa `typing` para anotaciones de tipo.

---

## Clase 4 — Metropolis-Hastings

**Tema:** Implementación manual del algoritmo M-H; diagnósticos de cadenas MCMC.

### Patrón: M-H from scratch

```python
def metropolis_for_target(x, alpha=1):
    y = np.random.uniform(x - alpha, x + alpha)
    y = x if (np.random.rand() > target(y) / target(x)) else y
    return y

# Iterar manualmente
trace = np.ones(T)
for t in range(1, T):
    trace[t] = metropolis_for_target(trace[t - 1])
```

### Diagnósticos de cadenas con ArviZ

```python
# Pasar dict de arrays (n_chains × n_draws) a arviz directamente
dict_chains = {"variable": np.zeros((n_chains, n_draws))}

az.plot_trace(dict_chains)
az.ess(dict_chains)          # Effective Sample Size
az.rhat(dict_chains)         # R-hat (convergencia entre cadenas)
az.mcse(dict_chains)         # Markov Chain Standard Error
az.plot_rank(dict_chains, kind="bars")
az.plot_ess(dict_chains, kind="local")
az.plot_ess(dict_chains, kind="quantile")
az.plot_mcse(dict_chains)
```

### M-H aplicado al modelo Beta-Binomial

```python
# Posterior kernel
prior = scipy.stats.beta(alpha, beta).pdf(theta)
likelihood = scipy.stats.bernoulli(theta).pmf(Y).prod()
prob = likelihood * prior

# Paso M-H
theta_can = scipy.stats.norm(theta, candidate_sd).rvs(1)
p_acceptance = post(theta_can, ...) / post(theta, ...)
```

---

## Clase 5 — Modelos Lineales I

**Tema:** Estimación de parámetros, regresión lineal simple y múltiple.

### Ajuste de distribución con PyMC

```python
with pymc.Model(coords={"n_obs": range(n)}) as model_fit:
    mu    = pymc.Normal("mu", mu=mean_obs, sigma=std_obs)
    sigma = pymc.HalfStudentT("sigma", nu=1, sigma=5)
    likelihood = pymc.Normal("likelihood", mu=mu, sigma=sigma, dims="n_obs")

    prior_samples = pymc.sample_prior_predictive(draws=1000)

# Condicionar después de construir el modelo
model_cond = pymc.observe(model_fit, {"likelihood": data})
with model_cond:
    trace = pymc.sample(chains=5)
    prior_samples.extend(trace)
    posterior = pymc.sample_posterior_predictive(prior_samples)
    prior_samples.extend(posterior)

az.plot_dist_comparison(prior_samples, var_names=["mu"])
```

### Modelo multi-categoría (penguins)

```python
categories = pd.Categorical(df["species"])
n_cats = categories.categories.shape[0]

with pymc.Model() as model_all_species:
    mu    = pymc.Normal("mu", mu=4000, sigma=1000, shape=n_cats)
    sigma = pymc.HalfStudentT("sigma", nu=100, sigma=2000, shape=n_cats)
    mass  = pymc.Normal("mass",
                         mu=mu[categories.codes],
                         sigma=sigma[categories.codes],
                         observed=df["body_mass_g"].to_numpy())
    trace = pymc.sample(chains=5)

az.plot_forest(trace, var_names="mu")
```

### Regresión lineal bayesiana

```python
with pymc.Model() as linear_model:
    x_data = pymc.Data("flipper_length", x)
    b0 = pymc.Normal("b0", mu=0, sigma=4000)
    b1 = pymc.Normal("b1", mu=0, sigma=4000)
    sigma = pymc.HalfStudentT("sigma", nu=100, sigma=2000)
    mu   = pymc.Deterministic("mu", b0 + b1 * x_data)
    mass = pymc.Normal("mass", mu=mu, sigma=sigma, observed=y)
    trace = pymc.sample(chains=5, return_inferencedata=True)
    ppc   = pymc.sample_posterior_predictive(trace)

# Extraer muestras de posterior para graficar incertidumbre
b0_samples = trace["posterior"]["b0"].stack(all_draws=["chain","draw"]).values.T
```

---

## Clase 6 — Modelos Lineales Continuación

**Tema:** Regresión logística, priors informativos, modelos pooled vs. jerárquicos.

### Regresión logística

```python
with pymc.Model() as logistic_model:
    b0 = pymc.Normal("b0", mu=0, sigma=10)
    b1 = pymc.Normal("b1", mu=0, sigma=10)
    exponent = b0 + pymc.math.dot(b1, x)
    theta = pymc.Deterministic("theta", pymc.math.sigmoid(exponent))
    prob  = pymc.Bernoulli("prob", p=theta, observed=y_binary)

    trace = pymc.sample(chains=5, cores=8, draws=5000,
                         target_accept=0.99, tune=2000)

# Decision boundary: x = -b0/b1
b0_post = np.concat(trace.posterior.b0.to_numpy())
b1_post = np.concat(trace.posterior.b1.to_numpy())
sns.kdeplot(-b0_post / b1_post)
```

### Modelo jerárquico (parcialmente pooled)

```python
with pymc.Model(coords=dict_coords) as hierarchical_model:
    hypersigma = pymc.HalfNormal("hypersigma", 10)          # hyperprior
    sigma = pymc.HalfNormal("sigma", hypersigma, shape=3)   # prior del grupo
    b1    = pymc.Normal("b1", 10, 20, shape=3)
    mu    = pymc.Deterministic("mu", b1[cat.codes] * x)
    pymc.Normal("sales", mu=mu, sigma=sigma[cat.codes], observed=y)
    trace = pymc.sample(chains=5, draws=2000, target_accept=0.95, tune=1000)

pymc.model_to_graphviz(hierarchical_model)   # visualización del grafo
```

### Funciones helper definidas en clase

```python
def get_bounds_df(x, y_obs, array_numpy, conf=0.99) -> pd.DataFrame:
    """Genera DataFrame con cuantiles para graficar bandas de confianza."""

def plot_result(df_plot, field_nominal="median") -> plt.Plot:
    """Grafica observaciones + estimado nominal + banda de confianza."""
```

---

## Clase 7 — Splines y Series de Tiempo

**Tema:** Descomposición de series de tiempo, modelos con estacionalidad, base de Fourier.

### Serie de tiempo con regresión lineal + indicadores de mes

```python
# Indicadores de mes como variables dummy (estacionalidad)
arr_seasonality = pd.get_dummies([x.month for x in df["date_month"]]).to_numpy().astype(np.float32)
x_time = np.linspace(0., 1., len(df))[..., None]

dict_coords = {"months": np.arange(12) + 1}

with pymc.Model(coords=dict_coords) as model_time_series:
    b0    = pymc.Normal("b0", mu=320, sigma=50)
    trend = pymc.Normal("trend", mu=0, sigma=10)
    month = pymc.Normal("month", sigma=5, dims="months")  # coef. estacional

    mu = pymc.Deterministic(
        "mu",
        b0 + pymc.math.dot(trend, x) + pymc.math.dot(month, arr_season.T)
    )
    sigma = pymc.HalfCauchy("sigma", beta=0.5)
    concentration = pymc.Normal("concentration", mu=mu, sigma=sigma, observed=y_train)

    trace = pymc.sample(chains=5, draws=3000, target_accept=0.99, tune=1000)
```

### Base de Fourier para estacionalidad

```python
def gen_basis(inds_month, n, period):
    """Genera base de Fourier (cos + sin) para n frecuencias."""
    out = 2 * np.pi * (np.arange(n) + 1) * inds_month[..., None] / period
    return np.concatenate([np.cos(out), np.sin(out)], axis=1)

# Ejemplo: 6 frecuencias, período anual en meses
inds = np.array([x.month - 1 for x in vec_dates_train])
basis = gen_basis(inds, 6, 12)   # shape: (T, 12)

with pymc.Model(coords={"months": np.arange(12)+1}) as model_fourier:
    b0, b1, b2, b3 = [pymc.Normal(f"b{i}", ...) for i in range(4)]
    fourier = pymc.Normal("fourier", mu=0, sigma=1, dims="months")
    mu = pymc.Deterministic(
        "mu",
        b0 + pymc.math.dot(b1, x) + pymc.math.dot(b2, x**2)
        + pymc.math.dot(b3, x**3) + pymc.math.dot(fourier, basis.T)
    )
```

---

## Resumen de distribuciones usadas como priors

| Distribución | Uso típico en las clases |
|---|---|
| `pymc.Normal(mu, sigma)` | Media (μ), coeficientes de regresión (betas), pesos de NN, Fourier |
| `pymc.HalfStudentT(nu, sigma)` | Desviación estándar (σ > 0) — robusta |
| `pymc.HalfNormal(sigma)` | σ > 0, alternativa más simple |
| `pymc.HalfCauchy(beta)` | σ > 0 con colas pesadas — series de tiempo |
| `pymc.Exponential(lam)` | σ > 0 — prior débil, clase 6 |
| `pymc.Bernoulli(p)` | Likelihood de clasificación binaria |
| `pymc.Flat(...)` | Distribución plana para predicción (sin prior) |

---

## Flujo estándar de inferencia (todas las clases)

```python
with pymc.Model(...) as model:
    # 1. Priors
    # 2. Deterministic (transformaciones)  ← pymc.Deterministic(...)
    # 3. Likelihood con observed=...

    prior = pymc.sample_prior_predictive(draws=1000)
    trace = pymc.sample(chains=5, draws=2000, target_accept=0.95, tune=1000)
    ppc   = pymc.sample_posterior_predictive(trace)

# Diagnósticos mínimos
az.plot_trace(trace)
az.rhat(trace)           # < 1.01 es bueno
az.ess(trace)            # > 400 por cadena es razonable
az.plot_ppc(ppc)         # comparar predicción vs datos reales
```

---

## `requirements.txt` (sin pandas / matplotlib / numpy)

```
pymc>=5.0
arviz>=0.17
pytensor>=2.18
seaborn>=0.13
scipy>=1.12
scikit-learn>=1.4
graphviz>=0.20        # para pymc.model_to_graphviz()
```

> **Nota**: `pytensor` es el backend de compilación simbólica de PyMC (reemplazó a Theano). En Mac puede requerir `pytensor.config.cxx = "/usr/bin/clang++"`.

---

## Archivos `.py` a copiar para mantener el estilo

### 1. `classes/class_3/pymc_nn.py` ← **el más importante**

Demuestra el estilo y formato completo del curso:

- **Constantes privadas** para todos los nombres de variables/dimensiones:
  ```python
  _DATA_NAME_X = "X_data"
  _DIM_HIDDEN_LAYER_1 = "hidden_layer_1"
  _VARNAME_WEIGHTS_INTO_1 = "weights_into_1"
  ```
- **Funciones públicas** que construyen y corren modelos PyMC:
  ```python
  def construct_nn(x_train, y_train, rng, batch_size=50, n_hidden=5) -> pymc.Model:
  def sample_posterior_predictive(x, y, trace, n_hidden=5) -> np.ndarray:
  ```
- **Funciones privadas** para partes internas del modelo (`_get_coords`, `_neural_net`)
- **Anotaciones de tipo** con `typing` (`Union`, `Dict`, `Tuple`, `List`)
- Docstrings cortos (una línea de descripción, sin bloques largos)

### 2. `classes/class_2/coin_flip.py` ← secundario

Versión más sencilla del mismo patrón (modelo PyMC básico en función). Útil como plantilla de punto de partida.

---

## Convenciones de estilo observadas en el curso

| Patrón | Ejemplo |
|---|---|
| Constantes de campo/nombre | `_FIELD_SPECIES = "species"` |
| Funciones de construcción de modelo | `construct_nn(...)` devuelve `pymc.Model` |
| Coordenadas del modelo | `dict_coords = {"dim_name": np.arange(n)}` |
| Extraer posterior | `trace.posterior["var"].stack(all_draws=["chain","draw"]).values.T` |
| Concatenar chains | `np.concat(trace.posterior.var.to_numpy())` |
| Plot style global | `plt.style.use("dark_background")` |
| Config ArviZ | `az.style.use("arviz-darkgrid")` |
| Retina display | `%config InlineBackend.figure_format = 'retina'` |
