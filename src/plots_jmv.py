"""
Visualizaciones del modelo C
"""

from __future__ import annotations

from typing import * 

import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from matplotlib.patches import Patch

def plot_varianzas(idata: az.InferenceData, figsize = (8, 4)) -> plt.Figure:
    """
    Barras horizontales de las sigma — jerarquía de fuentes de varianza.
    """
    post = idata.posterior
    sigma_vars = {
        "Tipo\n(civil/militar)": post["sigma_tipo"].values.flatten(),
        "Año\n(tendencia temporal)": post["sigma_anio"].values.flatten(),
        "Institución\n(dentro de tipo)": post["sigma_inst"].values.flatten(),
        "Estado\n(geografía)": post["sigma_estado"].values.flatten(),
    }
    medias = [v.mean() for v in sigma_vars.values()]
    errs = [v.std() for v in sigma_vars.values()]
    labels = list(sigma_vars.keys())
    colores = ["#dc2626", "#f59e0b", "#2563eb", "#16a34a"]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, medias, xerr = errs, color = colores, alpha = 0.85, 
                   error_kw = dict(elinewidth = 1.5, capsize = 4))
    ax.axvline(0, color = "black", lw = 0.8)
    for bar, m in zip(bars, medias):
        ax.text(m + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{m:.3f}", va="center", fontsize = 10, fontweight = "bold")
    ax.set_xlabel("σ (escala logit)", fontsize = 11)
    ax.set_title("Jerarquía de fuentes de varianza en el surtimiento", fontsize = 12)
    ax.invert_yaxis()
    fig.tight_layout()

    print("Interpretación de σ:")
    print(f"  σ_tipo    = {medias[0]:.3f} → diferencia civil/militar domina el sistema")
    print(f"  σ_anio    = {medias[1]:.3f} → variación año a año (COVID visible)")
    print(f"  σ_inst    = {medias[2]:.3f} → diferencias dentro de cada bloque")
    print(f"  σ_estado  = {medias[3]:.3f} → geografía: importante pero no dominante")

    return fig

def plot_brecha_civil_militar(post: Dict[str, Any], figsize=(12, 4)) -> plt.Figure:
    """
    Distribución posterior de theta por tipo + histograma de la brecha.
    """

    et_civil = post["eff_tipo"][:, post["tipos"].index("civil")]
    et_mil = post["eff_tipo"][:, post["tipos"].index("militar")]
    theta_civil = 1 / (1 + np.exp(-(post["beta0"] + et_civil)))
    theta_mil = 1 / (1 + np.exp(-(post["beta0"] + et_mil)))
    brecha = theta_civil - theta_mil
    p_civil_gana = (et_civil > et_mil).mean()

    fig, axes = plt.subplots(1, 2, figsize = figsize)
    for arr, label, color in [(theta_civil, "Civil", "#2563eb"), (theta_mil, "Militar", "#dc2626")]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        axes[0].axvspan(lo, hi, alpha = 0.15, color = color)
        axes[0].axvline(arr.mean(), color = color, lw = 2.5, label = f"{label}: {arr.mean():.3f}")
    axes[0].set_xlabel("θ (prob. surtimiento completo)")
    axes[0].set_title("Distribución posterior de θ por tipo")
    axes[0].legend()

    axes[1].hist(brecha, bins = 80, color = "#7c3aed", alpha = 0.75, density = True)
    axes[1].axvline(brecha.mean(), color = "black", lw = 2, label = f"Brecha media = {brecha.mean():.3f}")
    axes[1].axvline(0, color = "red", lw = 1.5, linestyle = "--", label = "Sin diferencia")
    axes[1].set_xlabel("θ_civil − θ_militar")
    axes[1].set_title(f"Brecha civil-militar\nP(civil > militar) = {p_civil_gana:.0%}")
    axes[1].legend()

    fig.tight_layout()

    return fig

def plot_ranking_instituciones(
    post: Dict[str, Any], colores_inst: Dict[str, str], figsize = (13, 4)) -> plt.Figure:
    """
    Forest plot: eff_inst (logit) y theta (probabilidad) por institución.
    """

    insts = post["insts"]
    medias = [post["eff_inst"][:, i].mean() for i in range(len(insts))]
    orden = np.argsort(medias)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for rank, idx in enumerate(orden):
        inst = insts[idx]
        color = colores_inst.get(inst, "#374151")
        eff = post["eff_inst"][:, idx]
        lo, hi = np.percentile(eff, [2.5, 97.5])
        axes[0].plot([lo, hi], [rank, rank], color = color, lw = 2.5)
        axes[0].scatter(eff.mean(), rank, color = color, s = 80, zorder = 5)

        theta_i = 1 / (1 + np.exp(-(post["beta0"] + eff)))
        lo_t, hi_t = np.percentile(theta_i, [2.5, 97.5])
        axes[1].plot([lo_t, hi_t], [rank, rank], color = color, lw = 2.5)
        axes[1].scatter(theta_i.mean(), rank, color = color, s = 80, zorder = 5)

    axes[0].axvline(0, color = "gray", lw = 1, linestyle = "--")
    axes[0].set_yticks(range(len(insts)))
    axes[0].set_yticklabels([insts[i] for i in orden])
    axes[0].set_xlabel("eff_inst (logit)")
    axes[0].set_title("Efectos por institución")

    axes[1].set_yticks(range(len(insts)))
    axes[1].set_yticklabels([insts[i] for i in orden])
    axes[1].set_xlabel("θ (probabilidad)")
    axes[1].set_title("θ por institución")

    fig.tight_layout()

    return fig

def plot_tendencia_anual(post: Dict[str, Any], figsize = (13, 4)) -> plt.Figure:
    """
    Efecto año en escala logit y theta nacional por año.
    """

    anios = post["anios"]
    ea = post["eff_anio"]
    medias = ea.mean(axis = 0)
    lo = np.percentile(ea, 2.5, axis = 0)
    hi = np.percentile(ea, 97.5, axis = 0)

    fig, axes = plt.subplots(1, 2, figsize = figsize)
    axes[0].fill_between(anios, lo, hi, alpha = 0.25, color = "#7c3aed")
    axes[0].plot(anios, medias, "o-", color = "#7c3aed", lw = 2.5, ms = 8)
    axes[0].axhline(0, color = "gray", lw = 1, linestyle = "--")
    axes[0].set_xlabel("Año")
    axes[0].set_ylabel("eff_anio (logit)")
    axes[0].set_title("Efecto año en escala logit")
    axes[0].set_xticks(anios)

    theta_anio = 1 / (1 + np.exp(-(post["beta0"][:, None] + ea)))
    axes[1].fill_between(
        anios,
        np.percentile(theta_anio, 2.5, axis = 0),
        np.percentile(theta_anio, 97.5, axis = 0),
        alpha = 0.2, color = "#0891b2",
    )
    axes[1].plot(anios, theta_anio.mean(axis = 0), "s-", color = "#0891b2", lw = 2.5, ms = 8)
    axes[1].set_xlabel("Año")
    axes[1].set_ylabel("θ (promedio nacional)")
    axes[1].set_title("θ nacional por año")
    axes[1].set_xticks(anios)

    fig.tight_layout()

    return fig

def plot_ranking_geografico(post: Dict[str, Any], figsize = (7, 11)) -> plt.Figure:
    """
    Forest plot de eff_estado, ordenado de peor a mejor.
    """

    estados = post["estados"]
    es = post["eff_estado"]
    medias = es.mean(axis=0)
    lo = np.percentile(es, 2.5, axis = 0)
    hi = np.percentile(es, 97.5, axis = 0)
    orden = np.argsort(medias)

    fig, ax = plt.subplots(figsize = figsize)
    for rank, idx in enumerate(orden):
        color = "#dc2626" if medias[idx] < 0 else "#16a34a"
        ax.plot([lo[idx], hi[idx]], [rank, rank], color = color, lw = 1.8, alpha = 0.7)
        ax.scatter(medias[idx], rank, color = color, s = 40, zorder = 5)

    ax.axvline(0, color = "black", lw = 1.2, linestyle = "--", label = "Promedio nacional")
    ax.set_yticks(range(len(estados)))
    ax.set_yticklabels([estados[i] for i in orden], fontsize = 8)
    ax.set_xlabel("eff_estado (logit)")
    ax.set_title("Ranking geográfico de surtimiento")
    ax.legend(handles = [
        Patch(color = "#dc2626", label = "Por debajo del promedio"),
        Patch(color = "#16a34a", label = "Por encima del promedio"),
    ], fontsize = 9, loc = "lower right")
    fig.tight_layout()

    return fig


def plot_escenarios_criticos(
    post: Dict[str, Any],
    estados: List[str],
    instituciones: List[str],
    anio: int,
    theta_posterior_fn: Callable,
    figsize = (9, 5),
) -> plt.Figure:
    """
    Heatmap de % de recetas incompletas para una combinación de estados x
    instituciones, en un año dado. `theta_posterior_fn` es analysis.theta_posterior.
    """

    matrix = np.zeros((len(estados), len(instituciones)))
    for i, est in enumerate(estados):
        for k, inst in enumerate(instituciones):
            theta = theta_posterior_fn(post, inst = inst, estado = est, anio = anio)
            matrix[i, k] = (1 - theta.mean()) * 100

    fig, ax = plt.subplots(figsize = figsize)
    im = ax.imshow(matrix, cmap = "Reds", vmin = 5, vmax = 25, aspect = "auto")
    ax.set_xticks(range(len(instituciones)))
    ax.set_xticklabels(instituciones, fontsize = 11)
    ax.set_yticks(range(len(estados)))
    ax.set_yticklabels(estados, fontsize = 11)
    for i in range(len(estados)):
        for k in range(len(instituciones)):
            ax.text(k, i, f"{matrix[i, k]:.1f}%", ha = "center", va = "center",
                     fontsize = 12, fontweight = "bold",
                     color = "white" if matrix[i, k] > 15 else "black")
    fig.colorbar(im, ax = ax, label = "% de pacientes con receta incompleta")
    ax.set_title(f"% pacientes sin medicación completa — año {anio}")
    fig.tight_layout()
    
    return fig

def plot_regression_bounds(
    post: Dict[str, Any],
    data: Dict[str, Any],
    theta_posterior_fn: Callable, 
    instituciones: List[str] = None,
    hdi: float = 0.94,
    figsize=(12, 5),
) -> plt.Figure:
    """
    Regression bounds plot bayesiano con bandas visibles.

    La banda HDI incluye eff_estado en el cálculo — captura la dispersión
    geográfica real del modelo (variación entre los 32 estados). Esto produce
    bandas visibles e interpretables: "rango donde cae el 94% de los estados
    para esta institución y año".
    """
    df = data["df_model"]
    anios = post["anios"]

    if instituciones is None:
        instituciones = post["insts"]

    colores_inst = {
        "ISSSTE": "#34d807", "IMSS": "#2563eb", "IMSS Bienestar": "#e9290b",
        "SEDENA": "#073de1", "SEMAR": "#ef4444",
    }

    alpha_tail = (1 - hdi) / 2 * 100

    fig, ax = plt.subplots(figsize=figsize)

    for inst in instituciones:
        color = colores_inst.get(inst, "#374151")
        i = post["insts"].index(inst)

        medias, los, his = [], [], []
        for k, anio in enumerate(anios):
            logit_mat = (
                post["beta0"][:, None]            # (S, 1)
                + post["eff_inst"][:, i, None]    # (S, 1)
                + post["eff_anio"][:, k, None]    # (S, 1)
                + post["eff_estado"]              # (S, n_estados)
            )
            theta_mat = 1 / (1 + np.exp(-logit_mat))  # (S, n_estados)
            theta_flat = theta_mat.flatten()

            medias.append(theta_flat.mean())
            los.append(np.percentile(theta_flat, alpha_tail))
            his.append(np.percentile(theta_flat, 100 - alpha_tail))

        ax.fill_between(anios, los, his, alpha=0.25, color=color)
        ax.plot(anios, medias, "o-", color=color, lw=2.5, ms=7, label=inst)

        obs = (
            df[df["institucion"] == inst]
            .groupby("anio")
            .apply(lambda x: x["surtidas"].sum() / x["total"].sum())
        )
        ax.scatter(obs.index, obs.values, color=color, s=60,
                   marker="x", zorder=5, linewidths=2)

    ax.set_xlabel("Año", fontsize=11)
    ax.set_ylabel("θ (prob. surtimiento completo)", fontsize=11)
    ax.set_title(
        f"Regresión logística bayesiana — banda de credibilidad {int(hdi*100)}%\n"
        "Media posterior por institución y año (× = promedio nacional observado)",
        fontsize=12,
    )
    ax.set_xticks(anios)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    return fig