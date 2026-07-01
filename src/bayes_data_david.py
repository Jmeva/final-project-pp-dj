"""
Modelo jerárquico Beta-Binomial.

"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes de campos (nombres de columnas en los parquets)
# ---------------------------------------------------------------------------

_FIELD_ESTADO         = "estado"
_FIELD_ANIO           = "anio"
_FIELD_INSTITUCION    = "institucion"
_FIELD_TIPO           = "tipo"
_FIELD_MES            = "mes_num"
_FIELD_NIVEL_ACCESO   = "nivel_acceso"
_FIELD_NIVEL_COMPLETO = "Completo"
_FIELD_SIN_DATOS      = "Sin datos"

# Columnas que devuelve load_agg_data
_FIELD_SURTIDAS = "surtidas"
_FIELD_TOTAL    = "total"
_FIELD_FOLIO    = "folio"
_FIELD_NEGADO   = "Negado"
_FIELD_PARCIAL  = "Parcial"

# # ---------------------------------------------------------------------------
# # Mapeo de valores especiales de `estado` → nombre de estado real
# #
# # Los registros del ISSSTE usan "ZONA NORTE / SUR / ..." para la Ciudad de
# # México y nombres de hospitales regionales para otros estados.
# # ---------------------------------------------------------------------------

# _ESTADO_NORM: Dict[str, str] = {
#     "ZONA NORTE":   "CDMX",
#     "ZONA SUR":     "CDMX",
#     "ZONA ORIENTE": "CDMX",
#     "ZONA PONIENTE":"CDMX",
#     "CENTRAL":      "CDMX",
#     # Hospitales regionales en la CDMX
#     "C.M.N. 20 DE NOVIEMBRE":                   "CDMX",
#     "H.R. 1 DE OCTUBRE":                        "CDMX",
#     "H.R. PRESIDENTE BENITO JUAREZ":            "CDMX",
#     "H.R. GRAL. IGNACIO ZARAGOZA":              "CDMX",
#     "H.R. PRESIDENTE ADOLFO LOPEZ MATEOS":      "CDMX",
#     # Hospitales regionales en otros estados
#     "H.R. MONTERREY":                            "NUEVO LEÓN",
#     "H.R. VALENTIN GOMEZ FARIAS":               "JALISCO",
#     "H.R. MERIDA":                               "YUCATÁN",
#     "H.R. PUEBLA":                               "PUEBLA",
#     "H.R. LEON":                                 "GUANAJUATO",
#     "H.R. MANUEL CARDENAS DE LA VEGA":          "SINALOA",
#     "HAE BICENTENARIO DE LA INDEPENDENCIA":     "HIDALGO",
#     "HAE CENTENARIO DE LA REVOLUCIÓN MEXICANA": "CHIHUAHUA",
#     "HAE MORELIA":                              "MICHOACÁN",
#     "HAE VERACRUZ,VER.":                        "VERACRUZ",
# }

# # Prefijos de filas de error SQL que llegan en el campo `estado`
# _PREFIJOS_ERROR = ("ERROR", "ORA-", "SP2-", "SQL")


# # ---------------------------------------------------------------------------
# # Función privada: limpieza de la columna estado
# # ---------------------------------------------------------------------------

# def _normalizar_estado(df: pd.DataFrame) -> pd.DataFrame:
#     """Mapea ZONAs / hospitales al nombre del estado; descarta errores SQL."""
#     df = df.copy()
#     # Reemplazar valores del diccionario
#     df[_FIELD_ESTADO] = df[_FIELD_ESTADO].replace(_ESTADO_NORM)
#     # Descartar filas con mensajes de error en el campo estado
#     mask_error = df[_FIELD_ESTADO].str.startswith(_PREFIJOS_ERROR, na=False)
#     return df[~mask_error].copy()


# # ---------------------------------------------------------------------------
# # Función pública 1: agregar datos desde DuckDB
# # ---------------------------------------------------------------------------

# def load_agg_data(con, patron: str, nivel: str = "linea") -> pd.DataFrame:
#     """
#     Agrega los parquets ISSSTE a nivel (estado, anio, mes_num).

#     Parámetros
#     ----------
#     con    : conexión DuckDB activa (duckdb.DuckDBPyConnection)
#     patron : glob que apunta a los parquets, p. ej. '.../issste_*.parquet'
#     nivel  : "linea" (default) o "folio"
#              - "linea" : cada fila del parquet es un experimento Bernoulli.
#                          n = líneas de medicamento, k = líneas surtidas completas.
#              - "folio" : cada receta es un experimento Bernoulli.
#                          n = folios únicos, k = folios donde TODOS los medicamentos
#                          fueron surtidos completos. Supuesto de independencia más
#                          defendible porque los folios pertenecen a pacientes distintos.

#     Devuelve
#     --------
#     DataFrame con columnas:
#         estado, anio, mes_num, surtidas (int), total (int)
#     Los valores de `estado` ya están normalizados y sin errores SQL.
#     """
#     if nivel == "folio":
#         df = con.execute(f"""
#             WITH folio_estado AS (
#                 SELECT
#                     {_FIELD_FOLIO},
#                     {_FIELD_ANIO},
#                     {_FIELD_MES},
#                     FIRST({_FIELD_ESTADO}) AS {_FIELD_ESTADO},
#                     MAX(CASE
#                         WHEN {_FIELD_NIVEL_ACCESO} IN ('{_FIELD_NEGADO}', '{_FIELD_PARCIAL}')
#                         THEN 1 ELSE 0
#                     END) AS tiene_problema
#                 FROM read_parquet('{patron}')
#                 WHERE {_FIELD_NIVEL_ACCESO} != '{_FIELD_SIN_DATOS}'
#                   AND {_FIELD_ESTADO} IS NOT NULL
#                 GROUP BY {_FIELD_FOLIO}, {_FIELD_ANIO}, {_FIELD_MES}
#             )
#             SELECT
#                 {_FIELD_ESTADO},
#                 {_FIELD_ANIO},
#                 {_FIELD_MES},
#                 SUM(1 - tiene_problema) AS {_FIELD_SURTIDAS},
#                 COUNT(*)                AS {_FIELD_TOTAL}
#             FROM folio_estado
#             GROUP BY {_FIELD_ESTADO}, {_FIELD_ANIO}, {_FIELD_MES}
#             ORDER BY {_FIELD_ESTADO}, {_FIELD_ANIO}, {_FIELD_MES}
#         """).df()
#     else:
#         df = con.execute(f"""
#             SELECT
#                 {_FIELD_ESTADO},
#                 {_FIELD_ANIO},
#                 {_FIELD_MES},
#                 COUNT(*) FILTER (WHERE {_FIELD_NIVEL_ACCESO} = '{_FIELD_NIVEL_COMPLETO}')
#                     AS {_FIELD_SURTIDAS},
#                 COUNT(*) AS {_FIELD_TOTAL}
#             FROM read_parquet('{patron}')
#             WHERE {_FIELD_NIVEL_ACCESO} != '{_FIELD_SIN_DATOS}'
#               AND {_FIELD_ESTADO} IS NOT NULL
#             GROUP BY {_FIELD_ESTADO}, {_FIELD_ANIO}, {_FIELD_MES}
#             ORDER BY {_FIELD_ESTADO}, {_FIELD_ANIO}, {_FIELD_MES}
#         """).df()

#     df = _normalizar_estado(df)

#     # Reagregar después de normalizar (múltiples valores → mismo estado)
#     df = (
#         df.groupby([_FIELD_ESTADO, _FIELD_ANIO, _FIELD_MES], as_index=False)
#         [[_FIELD_SURTIDAS, _FIELD_TOTAL]]
#         .sum()
#         .sort_values([_FIELD_ESTADO, _FIELD_ANIO, _FIELD_MES])
#         .reset_index(drop=True)
#     )
#     return df


# ---------------------------------------------------------------------------
# Función pública 2: construir inputs para PyMC
# ---------------------------------------------------------------------------

def build_model_inputs(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Colapsa el DataFrame al nivel (estado, anio) y construye los arrays
    de índices enteros y el dict de coordenadas para PyMC.

    Parámetros
    ----------
    df : salida de load_agg_data()

    Devuelve
    --------
    dict con:
        estado_idx  : np.ndarray int  — índice 0..n_estados-1, shape (N,)
        anio_idx    : np.ndarray int  — índice 0..6, shape (N,)
        n_obs       : np.ndarray int  — total de líneas, shape (N,)
        k_obs       : np.ndarray int  — líneas surtidas, shape (N,)
        theta_obs   : np.ndarray float — tasa observada k/n, shape (N,)
        coords      : dict             — para pymc.Model(coords=...)
        df_model    : pd.DataFrame     — tabla de nivel (estado, anio)
    """
    # Agregar meses → nivel (estado, anio)
    df_model = (
        df.groupby([_FIELD_ESTADO, _FIELD_ANIO], as_index=False)
        [[_FIELD_SURTIDAS, _FIELD_TOTAL]]
        .sum()
        .reset_index(drop=True)
    )

    # Categorías ordenadas
    estados = sorted(df_model[_FIELD_ESTADO].unique())
    anios   = sorted(df_model[_FIELD_ANIO].unique())

    estado_cat = pd.Categorical(df_model[_FIELD_ESTADO], categories=estados)
    anio_cat   = pd.Categorical(df_model[_FIELD_ANIO],   categories=anios)

    n_obs = df_model[_FIELD_TOTAL].to_numpy(dtype=int)
    k_obs = df_model[_FIELD_SURTIDAS].to_numpy(dtype=int)

    return {
        "estado_idx":  estado_cat.codes.copy(),
        "anio_idx":    anio_cat.codes.copy(),
        "n_obs":       n_obs,
        "k_obs":       k_obs,
        "theta_obs":   k_obs / n_obs,
        "coords": {
            "estados": estados,
            "anios":   anios,
        },
        "df_model": df_model,
    }

# ---------------------------------------------------------------------------
# Función pública 3: construir inputs para el Modelo C (logístico jerárquico,
# institución anidada en tipo civil/militar)
# ---------------------------------------------------------------------------

def build_model_inputs_c(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Construye los arrays de índices y el dict de coordenadas para el Modelo C:
    logit(theta) = beta0 + eff_estado[estado] + eff_anio[anio] + eff_inst[institucion]
    eff_inst = eff_tipo[tipo de la institución] + z_inst * sigma_inst

    Parámetros
    ----------
    df : DataFrame con columnas estado, anio, surtidas, total, institucion, tipo
         (salida de load_all() en load_data_david.py)

    Devuelve
    --------
    dict con:
        estado_idx     : np.ndarray int — índice 0..n_estados-1, shape (N,)
        anio_idx       : np.ndarray int — índice 0..n_anios-1, shape (N,)
        inst_idx       : np.ndarray int — índice 0..n_instituciones-1, shape (N,)
        inst_to_tipo   : np.ndarray int — índice de tipo (0/1) por institución, shape (n_instituciones,)
        n_obs          : np.ndarray int — total de recetas, shape (N,)
        k_obs          : np.ndarray int — recetas surtidas, shape (N,)
        theta_obs      : np.ndarray float — tasa observada k/n, shape (N,)
        coords         : dict — para pymc.Model(coords=...), incluye "estado", "anio", "institucion", "tipo"
        df_model       : pd.DataFrame — tabla de nivel (estado, anio, institucion)
    """
    df_model = df.reset_index(drop=True)

    estados       = sorted(df_model[_FIELD_ESTADO].unique())
    anios         = sorted(df_model[_FIELD_ANIO].unique())
    instituciones = sorted(df_model[_FIELD_INSTITUCION].unique())
    tipos         = sorted(df_model[_FIELD_TIPO].unique())

    estado_cat = pd.Categorical(df_model[_FIELD_ESTADO], categories=estados)
    anio_cat   = pd.Categorical(df_model[_FIELD_ANIO], categories=anios)
    inst_cat   = pd.Categorical(df_model[_FIELD_INSTITUCION], categories=instituciones)
    tipo_cat   = pd.Categorical(df_model[_FIELD_TIPO], categories=tipos)

    # Mapeo institución -> tipo (una fila por institución, en el mismo orden que `instituciones`)
    inst_to_tipo_map = (
        df_model.drop_duplicates(_FIELD_INSTITUCION)
        .set_index(_FIELD_INSTITUCION)[_FIELD_TIPO]
        .reindex(instituciones)
    )
    inst_to_tipo = pd.Categorical(inst_to_tipo_map, categories=tipos).codes.copy()

    n_obs = df_model[_FIELD_TOTAL].to_numpy(dtype=int)
    k_obs = df_model[_FIELD_SURTIDAS].to_numpy(dtype=int)

    return {
        "estado_idx":   estado_cat.codes.copy(),
        "anio_idx":     anio_cat.codes.copy(),
        "inst_idx":     inst_cat.codes.copy(),
        "tipo_idx":     tipo_cat.codes.copy(),
        "inst_to_tipo": inst_to_tipo,
        "n_obs":        n_obs,
        "k_obs":        k_obs,
        "theta_obs":    k_obs / n_obs,
        "coords": {
            "estado":       estados,
            "anio":         anios,
            "institucion":  instituciones,
            "tipo":         tipos,
        },
        "df_model": df_model,
    }
