"""
Extracción y resumen del posterior del modelo C. 

Todas las funciones reciben 'idata' (InferenceData con sample) o el dict aplanado de 
extract_posterior(), y devuelven DataFrames / dicts 
"""

from __future__ import annotations

from typing import * 

import numpy as np
import pandas as pd 
import arviz as az 

def _flatten(post, var: str, n: int) -> np.ndarray:
    """
    Aplanar (chain, draw, dim) -> (samples, n)

    Args:
        post (_type_) 
        var (str) 
        n (int)

    Returns:
        np.ndarray
    """

    return post[var].values.reshape(-1, n)

def extract_posterior(idata: az.InferenceData, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplana las muestras del posterior a arrays en 2D (samples, n_categorias)

    También funciona para el modelo A, solo con el detalle de que 'eff_tipo' no está en el posterior del 
    Modelo A así que se omite. 

    Args:
        idata (az.InferenceData): InferenceData con el sample (contiene beta0, eff_inst, 
            eff_anio, eff_estado; eff_tipo es opcional)
        data (Dict[str, Any]): dict del model_data.preparar_modelo_a(df) o preparar_modelo_c(df). 
            Se usa para las listas de categorías (coords)

    Returns:
        Dict[str, Any]: con beta0, eff_inst, eff_anio, eff_estado, e insts/estados/anios en formato
            de lista. Si el modelo tiene tipo, incluye además eff_tipo y los tipos en formato de lista
    """

    post = idata.posterior
    coords = data["coords"]

    result = {
        "beta0": post["beta0"].values.flatten(),
        "eff_inst": _flatten(post, "eff_inst", len(coords["institucion"])),
        "eff_anio": _flatten(post, "eff_anio", len(coords["anio"])),
        "eff_estado": _flatten(post, "eff_estado", len(coords["estado"])),
        "insts": list(coords["institucion"]),
        "estados": list(coords["estado"]),
        "anios": list(coords["anio"]),
    }

    if "eff_tipo" in post: 
        result["eff_tipo"] = _flatten(post, "eff_tipo", len(coords["tipo"]))
        result["tipos"] = list(coords["tipo"])

    return result

def _invlogit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def theta_posterior(
        post: Dict[str, Any],
        inst: Optional[str] = None, 
        estado: Optional[str] = None, 
        anio: Optional[int] = None,
) -> np.ndarray: 
    """
    Distribución posterior de theta para una combinación (inst, estado, anio). 
    Cualquier argumento omitido se promedia sobre su dimensión. 
    anio = None promedia sobre años (efecto nacional)
    inst/estado = None omite del logit (se quedan en cero)

    Funciona para modelo A y el modelo C

    Args:
        post (Dict[str, Any]): distribución posterior
        inst (Optional[str], optional): Defaults to None.
        estado (Optional[str], optional): Defaults to None.
        anio (Optional[int], optional): Defaults to None.

    Returns:
        np.ndarray
    """

    logit = post["beta0"].copy()

    if inst is not None: 
        logit = logit + post["eff_inst"][:, post["insts"].index(inst)]

    if estado is not None:
        logit = logit + post["eff_estado"][:, post["estados"].index(estado)]
    
    if anio is not None: 
        logit = logit + post["eff_anio"][:, post["anios"].index(anio)]
    
    else: 
        logit = logit + post["eff_anio"].mean(axis = 1)
    
    return _invlogit(logit)


def brecha_civil_militar(post: Dict[str, Any]) -> pd.DataFrame: 
    """
    Resumen de la brecha civil/militara: theta media e IC 95% por tipo. 
    
    Solo aplica para modelo C ya que requiere 'eff_tipo'/'tipos' en 'post'
    Lanza KeyError si se intenta aplicar al modelo A. 

    Args:
        post (Dict[str, Any])

    Returns:
        pd.DataFrame
    """

    et_civil = post["eff_tipo"][:, post["tipos"].index("civil")]
    et_mil = post["eff_tipo"][:, post["tipos"].index("militar")]
    theta_civil = _invlogit(post["beta0"] + et_civil)
    theta_mil = _invlogit(post["beta0"] + et_mil)
    brecha = theta_civil - theta_mil

    rows = []
    for tipo,theta in [("civil", theta_civil), ("militar", theta_mil)]:
        rows.append({
            "tipo": tipo,
            "theta_media": theta.mean(),
            "ic_2.5": np.percentile(theta, 2.5),
            "ic_97.5": np.percentile(theta, 97.5),
        })
    df = pd.DataFrame(rows)
    df.attrs["brecha_media_pp"] = brecha.mean()
    df.attrs["brecha_ic_2.5"] = np.percentile(brecha, 2.5)
    df.attrs["brecha_ic_97.5"] = np.percentile(brecha, 97.5)
    df.attrs["p_civil_gana"] = (et_civil > et_mil).mean()
    return df 

def ranking_instituciones(post: Dict[str, Any]) -> pd.DataFrame: 
    """
    Theta media e IC 95% por institución, ordenado de mejor a peor.

    Args:
        post (Dict[str, Any])

    Returns:
        pd.DataFrame
    """
    rows = []
    for i, inst in enumerate(post["insts"]):
        theta_i = _invlogit(post["beta0"] + post["eff_inst"][:, i])
        rows.append({
            "institucion": inst,
            "theta_media": theta_i.mean(),
            "ic_2.5": np.percentile(theta_i, 2.5), 
            "ic_97.5": np.percentile(theta_i, 97.5),
            "receta_incompleta": 1 - theta_i.mean(),
        })
    return (
        pd.DataFrame(rows).sort_values("theta_media", ascending = False).reset_index(drop = True))

def ranking_estados(post: Dict[str, Any]) -> pd.DataFrame: 
    """
    eff_estado medio e IC 95% por estado, ordenado de peor a mejor

    Args:
        post (Dict[str, Any])

    Returns:
        pd.DataFrame
    """
    rows = []
    for j, estado in enumerate(post["estados"]):
        eff = post["eff_estado"][:, j]
        rows.append({
            "estado": estado,
            "eff_estado_media": eff.mean(),
            "ic_2.5": np.percentile(eff, 2.5),
            "ic_97.5": np.percentile(eff, 97.5),
            "p_eff_negativo": (eff < 0).mean(),
        })
    return pd.DataFrame(rows).sort_values("eff_estado_media").reset_index(drop = True)

def estados_estructurales(post: Dict[str, Any], umbral: float = 0.95) -> pd.DataFrame:
    """
    Estados con p(eff_estado < 0) > umbral: problema estructural confirmado

    Args:
        post (Dict[str, Any])
        umbral (float, optional): Defaults to 0.95.

    Returns:
        pd.DataFrame
    """
    df = ranking_estados(post)
    return df[df["p_eff_negativo"] > umbral].reset_index(drop = True)

def escenario(post: Dict[str, Any], inst: str, estado: str, anio: int) -> Dict[str, float]:
    """
    Resumen puntual de un escenario (institución, estado, año).

    Args:
        post (Dict[str, Any])
        inst (str)
        estado (str)
        anio (int)

    Returns:
        Dict[str, float]
    """
    theta = theta_posterior(post, inst = inst, estado = estado, anio = anio)
    return {
        "institucion": inst, 
        "estado": estado,
        "anio": anio,
        "theta_media": theta.mean(),
        "ic_2.5": np.percentile(theta, 2.5),
        "ic_97.5": np.percentile(theta, 97.5),
        "p_receta_incompleta": 1 - theta.mean(),
    }

# def declaraciones_politica_publica(post: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Declaraciones con certeza posterior cuantificada

    Args:
        post (Dict[str, Any])

    Returns:
        List[Dict[str, str]]
    """
    brecha = brecha_civil_militar(post)
    estructurales = estados_estructurales(post)

    et_civil = post["eff_tipo"][:, post["tipos"].index("civil")]
    et_mil = post["eff_tipo"][:, post["tipos"].index("militar")]
    p_civil_gana = (et_civil > et_mil).mean()

    ei_issste = post["eff_inst"][:, post["insts"].index("ISSSTE")]
    ei_imss = post["eff_inst"][:, post["insts"].index("IMSS")]
    p_issste_gana = (ei_issste > ei_imss).mean()

    return [
        {
            "titulo": f"P(sistema civil > sistema militar) = {p_civil_gana:.0%}",
            "detalle": f"Brecha media de {brecha.attrs['brecha_media_pp']:.1%} puntos porcentuales.",
        },
        {
            "titulo": f"P(ISSSTE > IMSS en surtimiento) = {p_issste_gana:.0%}",
            "detalle": "El ISSSTE supera al IMSS con certeza total pese a ser menos discutido públicamente.",
        },
        {
            "titulo": (
                f"{len(estructurales)} estados con problema estructural "
                "confirmado (P(eff_estado<0) > 95%)"
            ),
            "detalle": (
                "IMSS, ISSSTE e IMSS Bienestar están simultáneamente por debajo "
                "del promedio nacional en estos estados."
            ),
        },
    ]