"""
Funções puras (sem dependência de PySide6 ou sklearn).

Inclui:
    - cálculos de features engenheiradas (BMI, MAP, pulse_pressure, bp_ratio, obesity)
    - um score de risco composto (diagnóstico de UI, não entra no modelo)
    - validação dos dados inseridos pelo usuário
    - montagem do vetor de features no formato/ordem esperados pelo scaler

O modelo treinado espera, nesta ordem (detectado de scaler.feature_names_in_):
    gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco,
    active, age_years, bmi, pulse_pressure, map, bp_ratio, obesity, risk_score
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Features engenheiradas (todas recebem primitivos e devolvem primitivos).
# ---------------------------------------------------------------------------

def calcular_bmi(weight_kg: float, height_cm: float) -> float:
    """Índice de massa corporal: peso (kg) / altura (m)²."""
    if height_cm is None or weight_kg is None:
        raise ValueError("Altura e peso são obrigatórios para calcular o IMC.")
    if height_cm <= 0:
        raise ValueError("Altura deve ser maior que zero.")
    if weight_kg <= 0:
        raise ValueError("Peso deve ser maior que zero.")
    height_m = height_cm / 100.0
    return weight_kg / (height_m * height_m)


def calcular_map(ap_hi: int, ap_lo: int) -> float:
    """Pressão arterial média: AP_diastólica + (AP_sistólica - AP_diastólica) / 3."""
    if ap_hi is None or ap_lo is None:
        raise ValueError("Pressões sistólica e diastólica são obrigatórias.")
    if ap_lo <= 0:
        raise ValueError("Pressão diastólica deve ser maior que zero.")
    if ap_hi <= ap_lo:
        raise ValueError("Pressão sistólica deve ser maior que a diastólica.")
    return ap_lo + (ap_hi - ap_lo) / 3.0


def calcular_bp_ratio(ap_hi: int, ap_lo: int) -> float:
    """Razão entre pressão sistólica e diastólica."""
    if ap_hi is None or ap_lo is None:
        raise ValueError("Pressões sistólica e diastólica são obrigatórias.")
    if ap_lo <= 0:
        raise ValueError("Pressão diastólica deve ser maior que zero.")
    return ap_hi / ap_lo


def calcular_pulse_pressure(ap_hi: int, ap_lo: int) -> int:
    """Pressão de pulso: AP_sistólica - AP_diastólica."""
    if ap_hi is None or ap_lo is None:
        raise ValueError("Pressões sistólica e diastólica são obrigatórias.")
    if ap_hi <= ap_lo:
        raise ValueError("Pressão sistólica deve ser maior que a diastólica.")
    return int(ap_hi - ap_lo)


def calcular_obesity(bmi: float) -> int:
    """Indicador binário de obesidade (1 se IMC >= 30, senão 0)."""
    return int(bmi >= 30.0)


# ---------------------------------------------------------------------------
# Score de risco composto (apenas diagnóstico na UI; não alimenta o modelo).
# ---------------------------------------------------------------------------

def calcular_risk_score(features: Mapping[str, object]) -> float:
    """
    Heurística: soma contribuições normalizadas de fatores de risco
    clássicos (colesterol, glicose, fumo, álcool, idade, IMC, MAP).

    Retorna um valor em [0, 1].
    """
    score = 0.0

    chol = features.get("cholesterol")
    if chol is not None:
        try:
            score += max(0.0, (float(chol) - 1.0) / 2.0) * 0.20
        except (TypeError, ValueError):
            pass

    gluc = features.get("gluc")
    if gluc is not None:
        try:
            score += max(0.0, (float(gluc) - 1.0) / 2.0) * 0.10
        except (TypeError, ValueError):
            pass

    if features.get("smoke"):
        score += 0.15
    if features.get("alco"):
        score += 0.05

    age = features.get("age_years")
    if age is not None:
        try:
            score += 0.10 if float(age) > 50 else 0.0
        except (TypeError, ValueError):
            pass

    bmi = features.get("bmi")
    if bmi is not None:
        try:
            score += 0.20 if float(bmi) > 30 else 0.0
        except (TypeError, ValueError):
            pass

    map_v = features.get("map")
    if map_v is not None:
        try:
            score += 0.20 if float(map_v) > 100 else 0.0
        except (TypeError, ValueError):
            pass

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Validação dos campos crus.
# ---------------------------------------------------------------------------

def _to_float(v) -> float:
    if isinstance(v, bool):
        # QCheckBox checked é True; tratar como 0/1 depois, não como float.
        raise TypeError("valor booleano não é numérico")
    return float(v)


def validar_dados(dados: Mapping[str, object]) -> None:
    """Levanta ValueError com mensagem em pt-BR se algum campo for inválido."""
    required = [
        "age_years", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol", "gluc",
        "smoke", "alco", "active",
    ]
    for key in required:
        if key not in dados or dados[key] is None:
            raise ValueError(f"Campo obrigatório ausente: {key}.")

    try:
        age = _to_float(dados["age_years"])
    except (TypeError, ValueError):
        raise ValueError("Idade deve ser um número.")
    if not (1 <= age <= 120):
        raise ValueError("Idade deve estar entre 1 e 120 anos.")

    try:
        gender = int(dados["gender"])
    except (TypeError, ValueError):
        raise ValueError("Gênero deve ser 1 (Feminino) ou 2 (Masculino).")
    if gender not in (1, 2):
        raise ValueError("Gênero deve ser 1 (Feminino) ou 2 (Masculino).")

    try:
        height = _to_float(dados["height"])
    except (TypeError, ValueError):
        raise ValueError("Altura deve ser um número.")
    if not (50 <= height <= 250):
        raise ValueError("Altura deve estar entre 50 e 250 cm.")

    try:
        weight = _to_float(dados["weight"])
    except (TypeError, ValueError):
        raise ValueError("Peso deve ser um número.")
    if not (20 <= weight <= 400):
        raise ValueError("Peso deve estar entre 20 e 400 kg.")

    try:
        ap_hi = int(dados["ap_hi"])
        ap_lo = int(dados["ap_lo"])
    except (TypeError, ValueError):
        raise ValueError("Pressões sistólica e diastólica devem ser inteiras.")
    if not (50 <= ap_hi <= 250):
        raise ValueError("Pressão sistólica deve estar entre 50 e 250 mmHg.")
    if not (30 <= ap_lo <= 200):
        raise ValueError("Pressão diastólica deve estar entre 30 e 200 mmHg.")
    if ap_hi <= ap_lo:
        raise ValueError("Pressão sistólica deve ser maior que a diastólica.")

    try:
        cholesterol = int(dados["cholesterol"])
    except (TypeError, ValueError):
        raise ValueError("Colesterol deve ser 1, 2 ou 3.")
    if cholesterol not in (1, 2, 3):
        raise ValueError("Colesterol deve ser 1 (Normal), 2 (Acima) ou 3 (Muito acima).")

    try:
        gluc = int(dados["gluc"])
    except (TypeError, ValueError):
        raise ValueError("Glicose deve ser 1, 2 ou 3.")
    if gluc not in (1, 2, 3):
        raise ValueError("Glicose deve ser 1 (Normal), 2 (Acima) ou 3 (Muito acima).")

    for key in ("smoke", "alco", "active"):
        v = dados[key]
        if not isinstance(v, (bool, int)) or int(bool(v)) not in (0, 1):
            raise ValueError(f"Campo {key} deve ser 0 ou 1.")


# ---------------------------------------------------------------------------
# Montagem do vetor de features na ordem esperada pelo scaler.
# ---------------------------------------------------------------------------

_ENGINEERED_COMPUTERS = {
    "bmi": lambda d: calcular_bmi(_to_float(d["weight"]), _to_float(d["height"])),
    "pulse_pressure": lambda d: calcular_pulse_pressure(int(d["ap_hi"]), int(d["ap_lo"])),
    "map": lambda d: calcular_map(int(d["ap_hi"]), int(d["ap_lo"])),
    "bp_ratio": lambda d: calcular_bp_ratio(int(d["ap_hi"]), int(d["ap_lo"])),
    "obesity": lambda d: calcular_obesity(
        _ENGINEERED_COMPUTERS["bmi"](d)
    ),
    "risk_score": lambda d: calcular_risk_score(
        {**d, "bmi": _ENGINEERED_COMPUTERS["bmi"](d),
            "map": _ENGINEERED_COMPUTERS["map"](d)}
    ),
}


def montar_vetor_features(dados: Mapping[str, object], feature_order) -> np.ndarray:
    """
    Constrói a matriz (1, n_features) com os valores na ordem exigida pelo scaler.

    `dados` deve conter os 11 campos crus do dataset Kaggle (age_years, gender,
    height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active). Os
    campos engenheirados são computados sob demanda. Valores faltantes viram
    NaN (o que seria problemático se o scaler não os imputou; manter
    `validar_dados` antes desta função é recomendado).
    """
    if feature_order is None or len(feature_order) == 0:
        raise ValueError("Lista de features vazia; verifique o scaler.")

    # Copia rasa para não mutar o dict do chamador.
    work = dict(dados)

    # Pré-computa valores engenheirados (se forem pedidos).
    needed_engineered = [name for name in feature_order if name in _ENGINEERED_COMPUTERS]
    cached = {}
    for name in needed_engineered:
        try:
            cached[name] = _ENGINEERED_COMPUTERS[name](work)
        except (KeyError, ValueError, TypeError):
            cached[name] = None
        work[name] = cached[name]

    # Monta a linha na ordem do scaler.
    row = []
    for name in feature_order:
        v = work.get(name, None)
        if v is None:
            row.append(np.nan)
            continue
        if isinstance(v, bool):
            row.append(float(int(v)))
        else:
            try:
                row.append(float(v))
            except (TypeError, ValueError):
                row.append(np.nan)

    arr = np.array([row], dtype=np.float64)
    if arr.shape[1] != len(feature_order):
        raise RuntimeError(
            f"Feature order length mismatch: expected {len(feature_order)}, got {arr.shape[1]}"
        )
    return arr
