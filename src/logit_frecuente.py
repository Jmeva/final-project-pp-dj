"""
Modelos logit frecuentistas (GLM binomial, liga logit) como baseline
de comparación para el Modelo A y el Modelo C bayesianos.

Los predictores son efectos fijos con dummies — sin pooling parcial.
Eso es exactamente lo que el modelo bayesiano jerárquico mejora.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_logit_a(df: pd.DataFrame) -> Any:
    """
    GLM binomial logit — Modelo A (solo instituciones civiles).
    logit(theta) = beta0 + efectos_fijos(estado) + efectos_fijos(anio)
                 + efectos_fijos(institucion)

    Parámetros
    ----------
    df : DataFrame completo; se filtra internamente a tipo == 'civil'.

    Devuelve
    --------
    Resultado de statsmodels GLM (.fit()), listo para .summary() o
    pasarlo a resumen_coeficientes().
    """

    df_civil = df[df["tipo"] == "civil"].copy()
    df_civil["prop"] = df_civil["surtidas"] / df_civil["total"]

    resultado = smf.glm(
        formula = "prop ~ C(estado) + C(anio) + C(institucion)",
        data = df_civil,
        family = sm.families.Binomial(),
        freq_weights = df_civil["total"],
    ).fit()

    return resultado


def fit_logit_c(df: pd.DataFrame) -> Any:
    """
    GLM binomial logit — Modelo C (civil + militar).
    logit(theta) = beta0 + efectos_fijos(estado) + efectos_fijos(anio)
                 + efectos_fijos(institucion)

    Nota: tipo e institución son colineales (cada institución pertenece a
    un único tipo), por eso se usa solo C(institucion) — los coeficientes
    de SEDENA/SEMAR vs. las civiles capturan implícitamente la brecha
    civil/militar.

    Parámetros
    ----------
    df : DataFrame completo con las 5 instituciones.

    Devuelve
    --------
    Resultado de statsmodels GLM (.fit()).
    """
    df_fit = df.copy()
    df_fit["prop"] = df_fit["surtidas"] / df_fit["total"]

    resultado = smf.glm(
        formula = "prop ~ C(estado) + C(anio) + C(institucion)",
        data = df_fit,
        family = sm.families.Binomial(),
        freq_weights = df_fit["total"],
    ).fit()

    return resultado


def resumen_coeficientes(resultado: Any) -> pd.DataFrame:
    """
    Extrae la tabla de coeficientes del resultado GLM en un DataFrame limpio.

    Devuelve columnas: variable, coef, ic_2.5, ic_97.5, p_valor, theta
    donde theta = invlogit(intercepto + coef) para interpretar en escala
    de probabilidad.
    """
    ic = resultado.conf_int()
    ic.columns = ["ic_2.5", "ic_97.5"]

    df_res = pd.DataFrame({
        "coef": resultado.params,
        "ic_2.5": ic["ic_2.5"],
        "ic_97.5": ic["ic_97.5"],
        "p_valor": resultado.pvalues,
    }).reset_index().rename(columns={"index": "variable"})

    # Añadir theta = invlogit(intercepto + coef) para escala de probabilidad
    intercept = resultado.params["Intercept"]
    df_res["theta"] = 1 / (1 + np.exp(-(intercept + df_res["coef"])))

    return df_res