"""
Carga y normalización de datos de surtimiento de recetas.

Tres fuentes:
  - ISSSTE      : CSV agregado (mensual → anual)
  - IMSS        : Excel hoja 'Recetas' (anual, 32 estados, 2019-2024*)
  - IMSS Bienestar: Excel 4 hojas wide format (anual, 20 estados, 2017-2024*)

* Datos de 2024 parciales (hasta abril).

Salida unificada: estado, anio, surtidas, total, institucion
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Normalización de nombres de estado
# ---------------------------------------------------------------------------

# ISSSTE usa estos nombres como referencia canónica.
# Los mapeos llevan las otras fuentes a ese estándar.

_NORM_IMSS: dict[str, str] = {
    "CIUDAD DE MEXICO": "CDMX",
    "ESTADO DE MEXICO": "MÉXICO",
    "MICHOACAN":        "MICHOACÁN",
    "NUEVO LEON":       "NUEVO LEÓN",
    "YUCATAN":          "YUCATÁN",
}

_NORM_IMSS_BW: dict[str, str] = {
    "COAHUILA DE ZARAGOZA":           "COAHUILA",
    "VERACRUZ DE IGNACIO DE LA LLAVE": "VERACRUZ",
    "MICHOACAN":                       "MICHOACÁN",
    "YUCATAN":                         "YUCATÁN",
    "MEXICO":                          "MÉXICO",
}

_COLS_OUT = ["estado", "anio", "surtidas", "total", "institucion"]

# ---------------------------------------------------------------------------
# ISSSTE
# ---------------------------------------------------------------------------

def load_issste(path: str | Path) -> pd.DataFrame:
    """
    Carga el CSV agregado del ISSSTE y colapsa meses → nivel anual.

    path : ruta a agg_folio_2018_2024.csv o agg_linea_2018_2024.csv
    """
    df = pd.read_csv(Path(path))
    df_anual = (
        df.groupby(["estado", "anio"], as_index=False)[["surtidas", "total"]]
        .sum()
    )
    df_anual[["surtidas", "total"]] = df_anual[["surtidas", "total"]].astype(int)
    df_anual["institucion"] = "ISSSTE"
    return df_anual[_COLS_OUT].reset_index(drop=True)


# ---------------------------------------------------------------------------
# IMSS
# ---------------------------------------------------------------------------

def load_imss(path: str | Path) -> pd.DataFrame:
    """
    Carga el Excel del IMSS (hoja 'Recetas').

    k = RECETAS COMPLETAMENTE SURTIDAS
    n = RECETAS PRESENTADAS
    """
    df = pd.read_excel(Path(path), sheet_name="Recetas")
    df = df.rename(columns={
        "AÑO":                           "anio",
        "ESTADO":                        "estado",
        "RECETAS COMPLETAMENTE SURTIDAS": "surtidas",
        "RECETAS PRESENTADAS":            "total",
    })[["estado", "anio", "surtidas", "total"]]
    df["estado"] = df["estado"].replace(_NORM_IMSS)
    df["institucion"] = "IMSS"
    return df[_COLS_OUT].reset_index(drop=True)


# ---------------------------------------------------------------------------
# IMSS Bienestar
# ---------------------------------------------------------------------------

def _parse_bw_sheet(xl: pd.ExcelFile, sheet: str, col_name: str) -> pd.DataFrame:
    """
    Parsea una hoja del Excel IMSS Bienestar.

    La fila de encabezado varía entre hojas; se localiza buscando la celda
    que contenga "ESTADO" en la segunda columna.
    Devuelve DataFrame largo con columnas: estado, anio, <col_name>.
    """
    raw = xl.parse(sheet, header=None)
    # Localizar la fila de encabezado (la que tiene "ESTADO" en col 1)
    header_idx = next(
        i for i, row in raw.iterrows() if row.iloc[1] == "ESTADO"
    )
    header_row = raw.iloc[header_idx].tolist()
    # Saltar encabezado; dropna limpia la fila vacía intermedia y el total final
    data = raw.iloc[header_idx + 1:].copy()
    data.columns = header_row
    data = data.dropna(subset=["ESTADO"])
    # Columnas de años: pueden ser int o float según si tienen NaN en esa col
    year_cols = [
        c for c in header_row
        if isinstance(c, (int, float)) and not pd.isna(c)
    ]
    df = data.melt(
        id_vars=["ESTADO"],
        value_vars=year_cols,
        var_name="anio",
        value_name=col_name,
    )
    df["anio"]   = df["anio"].astype(float).astype(int)
    df["ESTADO"] = df["ESTADO"].str.strip()
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0).astype(int)
    return df.rename(columns={"ESTADO": "estado"})[["estado", "anio", col_name]]


def load_imss_bienestar(path: str | Path) -> pd.DataFrame:
    """
    Carga el Excel de IMSS Bienestar.

    Numeral 1 = recetas totales presentadas
    Numeral 2 = recetas surtidas (completas)
    """
    xl = pd.ExcelFile(Path(path))
    total    = _parse_bw_sheet(xl, "Numeral 1", "total")
    surtidas = _parse_bw_sheet(xl, "Numeral 2", "surtidas")
    df = total.merge(surtidas, on=["estado", "anio"])
    df["estado"]      = df["estado"].replace(_NORM_IMSS_BW)
    df["institucion"] = "IMSS Bienestar"
    return df[_COLS_OUT].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Carga unificada
# ---------------------------------------------------------------------------

def load_all(
    data_dir: str | Path = "data",
    nivel_issste: str = "folio",
) -> pd.DataFrame:
    """
    Carga y combina los tres datasets en un DataFrame unificado.

    Parámetros
    ----------
    data_dir      : directorio donde viven los archivos de datos
    nivel_issste  : "folio" (default) o "linea" — qué CSV del ISSSTE usar

    Devuelve
    --------
    DataFrame con columnas: estado, anio, surtidas, total, institucion
    """
    data_dir = Path(data_dir)

    csv_issste = data_dir / f"agg_{nivel_issste}_2018_2024.csv"
    xls_imss   = data_dir / "IMSS_2019_ABRIL2024_ANUAL_SOLICITUD 330018024016694 ANEXO I.xlsx"
    xls_bw     = data_dir / "IMSS_BIENESTAR_2017_ABRIL2024_ANUAL_SOLICITUD 330018024016695 ANEXO I.xlsx"

    issste = load_issste(csv_issste)
    imss   = load_imss(xls_imss)
    bw     = load_imss_bienestar(xls_bw)

    return (
        pd.concat([issste, imss, bw], ignore_index=True)
        .sort_values(["institucion", "estado", "anio"])
        .reset_index(drop=True)
    )
