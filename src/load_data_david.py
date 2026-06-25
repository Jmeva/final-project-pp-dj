"""
Carga y normalización de datos de surtimiento de recetas.

Cinco fuentes:
  - ISSSTE          : CSV agregado (mensual → anual)
  - IMSS            : Excel hoja 'Recetas' (anual, 32 estados, 2019-2024*)
  - IMSS Bienestar  : Excel 4 hojas wide format (anual, 20 estados, 2017-2024*)
  - SEDENA          : Excel por hospital (anual, 24 estados, 2021-2023)
  - SEMAR           : Excel pivote ancho, hoja 'DH' (anual, 18 estados, 2019-2023)

* Datos de 2024 parciales (hasta abril).

Salida unificada: estado, anio, surtidas, total, institucion, tipo
"""
from __future__ import annotations

import re
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

_EXCLUIR_SEDENA = {"ESCALONES SANITARIOS", "RECETAS EXPEDIDAS", "TOTAL", "TOTAL GENERAL"}

_SEDENA_LAYOUT = {
    "2021": {"col_hospital": 3, "col_expedidas": 4, "col_surtidas": 6},
    "2022": {"col_hospital": 1, "col_expedidas": 2, "col_surtidas": 3},
    "2023": {"col_hospital": 1, "col_expedidas": 2, "col_surtidas": 3},
}

_ABREV_ESTADO_SEDENA: dict[str, str] = {
    "SIN.": "SINALOA", "CHIH.": "CHIHUAHUA", "JAL.": "JALISCO", "GTO.": "GUANAJUATO",
    "MEX.": "MÉXICO", "CD. MÉX.": "CDMX", "D.F.": "CDMX", "MOR.": "MORELOS",
    "ZAC.": "ZACATECAS", "OAX.": "OAXACA", "VER.": "VERACRUZ", "TAB.": "TABASCO",
    "CHIS.": "CHIAPAS", "TAMPS.": "TAMAULIPAS", "S.L.P.": "SAN LUIS POTOSI",
    "GRO.": "GUERRERO", "DGO.": "DURANGO", "MICH.": "MICHOACÁN",
    "Q. ROO.": "QUINTANA ROO", "B.C.": "BAJA CALIFORNIA", "SON.": "SONORA",
    "COAH.": "COAHUILA", "YUC": "YUCATÁN", "PUE": "PUEBLA",
}

_MAPEO_MANUAL_SEDENA: dict[str, str] = {
    "CENTRO DE REHABILITACIÓN INFANTIL": "CDMX",
    "H. M. E. DE LA MUJ. Y NEONATOLOGIA.": "CDMX",
    "H.M.Z CONTITUYENTES": "CDMX",  # typo de "CONSTITUYENTES"
    "HOSP. CENTRAL MILITAR": "CDMX",
    "U. M. C. E. DE LA S.D.N.": "CDMX",
    "U.M.C.E. LEONES, TACUBA": "CDMX",
    "UNIDAD ESP. MEDICAS.": "CDMX",
    "UNIDAD ESP. ODONTOLOGICAS.": "CDMX",
    "CHEMP. LOS PINOS, D.F. H.M.Z. CONSTITUYENTES": "CDMX",  # captura sucia: dos hospitales pegados
    "HOSP.. MIL. DE ESPECIALIDADES DE  MONTERREY": "NUEVO LEÓN",
    "H. M. R. DE EL CIPRES.": "BAJA CALIFORNIA",
    "U. M. C. E. TECAMACHALCO, D.F.  (ORIENTAL, PUEBLA).": "PUEBLA",
}

_EXCLUIR_ESTADO_SEMAR = {
    "ENTIDAD FEDERATIVA", "TOTAL PRIMER NIVEL", "TOTAL SEGUNDO NIVEL",
    "TOTAL TERCER NIVEL", "TOTAL GLOBAL",
}

_NORM_SEMAR: dict[str, str] = {
    "MICHOACAN": "MICHOACÁN",
    "YUCATAN": "YUCATÁN",
}

_ANIO_MIN_SEMAR, _ANIO_MAX_SEMAR = 2019, 2023

_TIPO_POR_INSTITUCION: dict[str, str] = {
    "IMSS": "civil", "ISSSTE": "civil", "IMSS Bienestar": "civil",
    "SEDENA": "militar", "SEMAR": "militar",
}

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
# SEDENA
# ---------------------------------------------------------------------------

def _limpiar_nombre_hospital(nombre: str) -> str:
    """Quita sufijo de número romano al final (con o sin 'RM'), ej. ' I RM', ' XII'."""
    return re.sub(r"\s*[IVXivx]+\s*(RM|-A)?\s*$", "", nombre.strip()).strip()


def _extraer_estado_sedena(nombre: str) -> str | None:
    for abrev, estado in _ABREV_ESTADO_SEDENA.items():
        if nombre.upper().rstrip(".").endswith(abrev.rstrip(".")):
            return estado
    return None


def _estado_de_hospital(nombre: str) -> str | None:
    limpio = _limpiar_nombre_hospital(nombre)
    return _extraer_estado_sedena(limpio) or _MAPEO_MANUAL_SEDENA.get(limpio)


def _parse_sedena_sheet(xl: pd.ExcelFile, hoja: str) -> pd.DataFrame:
    cfg = _SEDENA_LAYOUT[hoja]
    raw = xl.parse(hoja, header=None)
    data = raw.copy()
    data["hospital_raw"] = data[cfg["col_hospital"]]
    data = data[data["hospital_raw"].apply(lambda x: isinstance(x, str))].copy()
    data["hospital"] = data["hospital_raw"].apply(_limpiar_nombre_hospital)
    data = data[~data["hospital"].str.upper().isin(_EXCLUIR_SEDENA) & (data["hospital"] != "")]
    data["total"] = pd.to_numeric(data[cfg["col_expedidas"]], errors="coerce")
    data["surtidas"] = pd.to_numeric(data[cfg["col_surtidas"]], errors="coerce")
    data["anio"] = int(hoja)
    return data[["hospital", "anio", "surtidas", "total"]].dropna()


def load_sedena(path: str | Path) -> pd.DataFrame:
    """
    Carga el Excel de SEDENA (hojas 2021-2023; 2024 se descarta por estar incompleto).

    Los datos vienen por hospital, no por estado; se mapean con un catálogo
    manual + reglas de abreviatura embebida en el nombre del hospital.
    """
    xl = pd.ExcelFile(Path(path))
    partes = [_parse_sedena_sheet(xl, hoja) for hoja in ["2021", "2022", "2023"]]
    df = pd.concat(partes, ignore_index=True)
    df["estado"] = df["hospital"].apply(_estado_de_hospital)
    if df["estado"].isna().any():
        faltantes = df[df["estado"].isna()]["hospital"].unique()
        raise ValueError(f"Hospitales SEDENA sin mapeo a estado: {faltantes}")
    df_anual = df.groupby(["estado", "anio"], as_index=False)[["surtidas", "total"]].sum()
    df_anual[["surtidas", "total"]] = df_anual[["surtidas", "total"]].astype(int)
    df_anual["institucion"] = "SEDENA"
    return df_anual[_COLS_OUT].reset_index(drop=True)


# ---------------------------------------------------------------------------
# SEMAR
# ---------------------------------------------------------------------------

def _localizar_columnas_total_semar(df_raw: pd.DataFrame) -> dict[str, int]:
    """
    Por cada estado, ubica la columna con su total anual agregado.

    Si el estado tiene más de un nivel de atención, usa la columna con
    nivel == 'TOTAL' (ya suma todos los niveles); si solo tiene un nivel,
    no existe esa columna y se usa su única columna 'TOTAL ANUAL'.
    """
    fila_estado = df_raw.iloc[6].ffill()
    fila_nivel = df_raw.iloc[7]
    fila_mes = df_raw.iloc[8]

    estados = [e for e in fila_estado.dropna().unique() if e not in _EXCLUIR_ESTADO_SEMAR]

    col_total_por_estado = {}
    for estado in estados:
        cols = [c for c in range(df_raw.shape[1]) if fila_estado[c] == estado]
        col_nivel_total = [
            c for c in cols
            if isinstance(fila_nivel[c], str) and fila_nivel[c].strip().upper() == "TOTAL"
        ]
        if col_nivel_total:
            col_total_por_estado[estado] = col_nivel_total[0]
        else:
            cols_anual = [
                c for c in cols
                if isinstance(fila_mes[c], str) and fila_mes[c].strip().upper() == "TOTAL ANUAL"
            ]
            col_total_por_estado[estado] = cols_anual[0]

    return col_total_por_estado


def load_semar(path: str | Path, hoja: str = "DH") -> pd.DataFrame:
    """
    Carga el Excel de SEMAR (hoja 'DH'), ventana 2019-2023.

    hoja 'DH' = derechohabientes en general (fuente principal, confirmado
    empíricamente como superset de 'MA').
    """
    raw = pd.read_excel(Path(path), sheet_name=hoja, header=None)
    col_total_por_estado = _localizar_columnas_total_semar(raw)

    registros = []
    fila = 9
    while fila < len(raw):
        anio = raw.iloc[fila, 0]
        if pd.isna(anio) or not isinstance(anio, (int, float)):
            break
        anio = int(anio)
        if _ANIO_MIN_SEMAR <= anio <= _ANIO_MAX_SEMAR:
            for estado, col in col_total_por_estado.items():
                registros.append({
                    "estado": estado,
                    "anio": anio,
                    "surtidas": raw.iloc[fila, col],
                    "total": raw.iloc[fila + 3, col],
                })
        fila += 6

    df = pd.DataFrame(registros)
    df["estado"] = df["estado"].replace(_NORM_SEMAR)
    df[["surtidas", "total"]] = df[["surtidas", "total"]].round().astype(int)
    df["institucion"] = "SEMAR"
    return df[_COLS_OUT].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Carga unificada
# ---------------------------------------------------------------------------

def load_all(
    data_dir: str | Path = "data",
    nivel_issste: str = "folio",
) -> pd.DataFrame:
    """
    Carga y combina los cinco datasets en un DataFrame unificado.

    Parámetros
    ----------
    data_dir      : directorio donde viven los archivos de datos
    nivel_issste  : "folio" (default) o "linea" — qué CSV del ISSSTE usar

    Devuelve
    --------
    DataFrame con columnas: estado, anio, surtidas, total, institucion, tipo
    """
    data_dir = Path(data_dir)

    csv_issste = data_dir / f"agg_{nivel_issste}_2018_2024.csv"
    xls_imss   = data_dir / "IMSS_2019_ABRIL2024_ANUAL_SOLICITUD 330018024016694 ANEXO I.xlsx"
    xls_bw     = data_dir / "IMSS_BIENESTAR_2017_ABRIL2024_ANUAL_SOLICITUD 330018024016695 ANEXO I.xlsx"
    xls_sedena = data_dir / "SEDENA_2021_ABRIL2024_ANEXO FOLIO 330026424001543.xlsx"
    xls_semar  = data_dir / "SEMAR_2017_ABRIL2024_ANEXO.xlsx"

    issste = load_issste(csv_issste)
    imss   = load_imss(xls_imss)
    bw     = load_imss_bienestar(xls_bw)
    sedena = load_sedena(xls_sedena)
    semar  = load_semar(xls_semar)

    df = pd.concat([issste, imss, bw, sedena, semar], ignore_index=True)
    df["tipo"] = df["institucion"].map(_TIPO_POR_INSTITUCION)

    return df.sort_values(["institucion", "estado", "anio"]).reset_index(drop=True)
