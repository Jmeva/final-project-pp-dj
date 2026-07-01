"""
Preparación de inputs para el Modelo C (civil / militar anidado). 
"""

from __future__ import annotations

from typing import * 

import numpy as np
import pandas as pd 

from bayes_data_david import build_model_inputs_c 

def preparar_modelo_c(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Construye todos los inputs necesarios para el Modelo C: 
    logit(theta) = beta0 + eff_estado[estado] + eff_anio[anio] + eff_tipo[tipo] + dev_inst[institucion]

    dev_inst = ZeroSumNormal por bloque (civil / militar). 

    Args:
        df (pd.DataFrame): DataFrame con columnas estado, anio, surtidas, total, institución, tipo
            (salida de load_data_david.load_all())

    Returns:
        Dict[str, Any]: Set de build_model_inputs_c(df), con: 
            mask_civil: np.ndarray bool, shape (n_instituciones, )
            mask_mil: np.ndarray bool, shape (n_instituciones, )
        coords incluye "inst_civil" e "inst_mil", en el mismo orden que se filtran
        que coords["institución"], para pymc.Model(coords = ...)
    """
    data = build_model_inputs_c(df)

    instituciones = data["coords"]["institucion"]
    tipos = data["coords"]["tipo"]
    inst_to_tipo = data["inst_to_tipo"]

    mask_civil = np.array([tipos[t] == "civil" for t in inst_to_tipo])
    mask_mil = np.array([tipos[t] == "militar" for t in inst_to_tipo])

    inst_civil = [inst for inst, m in zip(instituciones, mask_civil) if m]
    inst_mil = [inst for inst, m in zip(instituciones, mask_mil) if m]

    data["mask_civil"] = mask_civil
    data["mask_mil"] = mask_mil
    data["coords"] = {
        **data["coords"],
        "inst_civil":inst_civil,
        "inst_mil": inst_mil,
    }
    return data

def preparar_modelo_a(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Construye los inputs del modelo A: solo instituciones civiles. 
    logit(theta) = beta0 + eff_inst[inst] + eff_anio[anio] + eff_estado[estado]

    Args:
        df (pd.DataFrame): DataFrame (con columna 'tipo') que se filtra internamente a 
            df['tipo'] == 'civil'

    Returns:
        Dict[str, Any]: salida del build_model_inputs_c únicamente aplicado al subconjunto civil.
    """

    df_civil = df[df["tipo"] == "civil"].copy()

    return build_model_inputs_c(df_civil)