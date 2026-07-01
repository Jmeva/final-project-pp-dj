"""
Especificación de los modelos PyMC: Modelo A (solo instituciones civiles) y
Modelo C (civil + militar, institución anidada en tipo).
"""

from __future__ import annotations

from typing import * 

import numpy as np
import pymc as pymc
import pytensor.tensor as ptt 

def build_modelo_a(data: Dict[str, Any], p_prior: float = 0.92) -> pymc.Model:
    """
    Modelo A: solo instituciones civiles. 
    logit(theta) = beta0 + eff_inst[inst] + eff_anio[anio] + eff_estado[estado]

    Args:
        data (Dict[str, Any]): salida de model_data.preparar_modelo_a(df)
        p_prior (float, optional): nivel de abasto promedio esperado a priori (default 0.92 fundado en nota oficial del gobierno federal)

    Returns:
        pymc.Model: modelo usando pymc sin samplear
    """

    coords = data["coords"]
    with pymc.Model(coords = coords) as modelo_A:
        beta0 = pymc.Normal("beta0", mu = np.log(p_prior / (1 - p_prior)), sigma = 0.5)

        sigma_inst = pymc.HalfNormal("sigma_inst", sigma = 0.5)
        eff_inst = pymc.ZeroSumNormal("eff_inst", sigma = sigma_inst, dims = "institucion")

        sigma_anio = pymc.HalfNormal("sigma_anio", sigma = 0.3)
        eff_anio = pymc.ZeroSumNormal("eff_anio", sigma = sigma_anio, dims = "anio")

        sigma_estado = pymc.HalfNormal("sigma_estado", sigma = 0.5)
        eff_estado = pymc.ZeroSumNormal("eff_estado", sigma = sigma_estado, dims = "estado")

        logit_theta = (
            beta0 + eff_inst[data["inst_idx"]] + eff_anio[data["anio_idx"]] + eff_estado[data["estado_idx"]]
        )

        theta = pymc.Deterministic("theta", pymc.math.invlogit(logit_theta))
        pymc.Binomial("k_obs", n = data["n_obs"], p = theta, observed = data["k_obs"])

        return modelo_A
    
def build_modelo_c(data: Dict[str, Any], p_prior: float = 0.92) -> pymc.Model:
    """
    Modelo C: civil + militar, institución anidada en tipo. 
    logit(theta) = beta0 + eff_estado[estado] + eff_anio[anio] + eff_tipo[tipo] + dev_inst[institucion]

    dev_inst = Aplicar ZeroSumNormal por bloque (dev_civil suma 0 dentro de civiles y dev_mil suma 0 dentro de militares)
        esto para mantener identificabilidad. 

    Args:
        data (Dict[str, Any]): salida de model_data.preparar_modelo_c(df)
        p_prior (float, optional): nivel de abasto promedio esperado a priori (default 0.92 fundamentado en comunicado del gobierno federal)

    Returns:
        pymc.Model: modelo sin samplear 
    """

    coords = data["coords"]
    mask_civil = data["mask_civil"]
    mask_mil = data["mask_mil"]
    n_inst = len(coords["institucion"])

    with pymc.Model(coords = coords) as modelo_C:
        beta0 = pymc.Normal("beta0", mu = np.log(p_prior / (1 - p_prior)), sigma = 0.5)

        sigma_estado = pymc.HalfNormal("sigma_estado", sigma = 0.5)
        eff_estado = pymc.ZeroSumNormal("eff_estado", sigma = sigma_estado, dims = "estado")

        sigma_anio = pymc.HalfNormal("sigma_anio", sigma = 0.3)
        eff_anio = pymc.ZeroSumNormal("eff_anio", sigma = sigma_anio, dims = "anio")

        sigma_tipo = pymc.HalfNormal("sigma_tipo", sigma = 0.5)
        eff_tipo = pymc.ZeroSumNormal("eff_tipo", sigma = sigma_tipo, dims = "tipo")

        sigma_inst = pymc.HalfNormal("sigma_inst", sigma = 0.3)
        dev_civil = pymc.ZeroSumNormal("dev_civil", sigma = sigma_inst, dims = "inst_civil")
        dev_mil = pymc.ZeroSumNormal("dev_mil", sigma = sigma_inst, dims = "inst_mil")

        # Ensamblar dev_civil / dev_mil en el orden de coords["institucion"]

        dev_dentro = ptt.zeros(n_inst)
        dev_dentro = ptt.set_subtensor(dev_dentro[np.where(mask_civil)[0]], dev_civil)
        dev_dentro = ptt.set_subtensor(dev_dentro[np.where(mask_mil)[0]], dev_mil)

        eff_inst = pymc.Deterministic("eff_inst", eff_tipo[data["inst_to_tipo"]] + dev_dentro, dims = "institucion")

        logit_theta = (beta0 + eff_estado[data["estado_idx"]] + eff_anio[data["anio_idx"]] + eff_inst[data["inst_idx"]])

        theta = pymc.Deterministic("theta", pymc.math.invlogit(logit_theta))
        
        pymc.Binomial("k_obs", n = data["n_obs"], p = theta, observed = data["k_obs"])

        return modelo_C