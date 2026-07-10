"""
Wrapper ao redor dos modelos .pkl pré-treinados (LR, RF, SVM).

Cada modelo está em `Modelos/<MODELO>/` acompanhado de `scaler.pkl` e
`pca.pkl`. As três pastas têm cópias idênticas dos mesmos scaler/PCA,
então carregamos uma única vez (de LR/) e reutilizamos para os três
modelos.

Pipeline: scaler (17 features) -> PCA (10 componentes) -> modelo (10 features).

A ordem das colunas é detectada em tempo de execução a partir de
`scaler.feature_names_in_`, com fallback para os modelos e, em último
caso, para a lista canônica de 17 colunas do dataset Kaggle.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

# Ordem canônica usada como último fallback. Inclui engineered features.
_KAGGLE_FALLBACK_ORDER: List[str] = [
    "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
    "age_years", "bmi", "pulse_pressure", "map", "bp_ratio",
    "obesity", "risk_score",
]


class Predictor:
    """Carrega LR, RF e SVM e expõe `predict_*` e `predict_all`."""

    def __init__(self, modelos_dir):
        self.modelos_dir = Path(modelos_dir)
        if not self.modelos_dir.exists():
            raise FileNotFoundError(
                f"Diretório de modelos não encontrado: {self.modelos_dir}"
            )

        # Suprime warnings de mismatch de versão do sklearn (modelos antigos
        # mas ainda compatíveis na prática).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.scaler = joblib.load(self.modelos_dir / "LR" / "scaler.pkl")
            self.pca = joblib.load(self.modelos_dir / "LR" / "pca.pkl")
            self.lr = joblib.load(self.modelos_dir / "LR" / "modelo_lr.pkl")
            self.rf = joblib.load(self.modelos_dir / "RF" / "modelo_rf.pkl")
            self.svm = joblib.load(self.modelos_dir / "SVM" / "modelo_svm.pkl")

        # Sanity check básico.
        if self.scaler.n_features_in_ != self.pca.n_features_in_:
            raise RuntimeError(
                "Scaler e PCA esperam número diferente de features "
                f"({self.scaler.n_features_in_} vs {self.pca.n_features_in_})."
            )

        # Detecta a ordem das colunas.
        self.feature_order: List[str] = self._detect_feature_order()
        if len(self.feature_order) != self.scaler.n_features_in_:
            raise RuntimeError(
                "Não foi possível determinar a ordem das features do scaler "
                f"(precisa de {self.scaler.n_features_in_}, "
                f"detectei {len(self.feature_order)})."
            )

        # Índice da classe "positiva" (doente = 1).
        self._positive_idx = self._detect_positive_index(self.lr)

        # Indica se cada modelo precisa do passo de PCA.
        self._lr_pca = self._needs_pca(self.lr)
        self._rf_pca = self._needs_pca(self.rf)
        self._svm_pca = self._needs_pca(self.svm)

    # ------------------------------------------------------------------ utils

    def _detect_feature_order(self) -> List[str]:
        # 1) scaler (autoritativo — é o primeiro transformador).
        names = getattr(self.scaler, "feature_names_in_", None)
        if names is not None:
            return [str(n) for n in names]

        # 2) qualquer modelo.
        for m in (self.lr, self.rf, self.svm):
            names = getattr(m, "feature_names_in_", None)
            if names is not None:
                return [str(n) for n in names]

        # 3) Fallback: primeiros n da lista canônica.
        n = self.scaler.n_features_in_
        return _KAGGLE_FALLBACK_ORDER[:n]

    def _detect_positive_index(self, model) -> int:
        classes = getattr(model, "classes_", None)
        if classes is None or len(classes) < 2:
            return 1
        # Pega o índice da classe de maior valor (em geral 1).
        return int(np.argmax(classes))

    def _needs_pca(self, model) -> bool:
        n_in = getattr(model, "n_features_in_", None)
        if n_in is None:
            return True
        # Se o modelo espera o número de componentes do PCA, aplica PCA.
        return n_in == self.pca.n_components_

    def _preprocess(self, features: np.ndarray, model) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Preserva os nomes das colunas esperados pelo scaler
        X = pd.DataFrame(features, columns=self.feature_order)

        X = self.scaler.transform(X)

        if self._needs_pca(model):
            X = self.pca.transform(X)

        return X

    def _positive_proba(self, model, X: np.ndarray) -> float:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return float(proba[0, self._positive_idx])
        # Fallback: sigmoid de decision_function (raro para os modelos incluídos).
        if hasattr(model, "decision_function"):
            d = float(model.decision_function(X)[0])
            return 1.0 / (1.0 + math.exp(-d))
        raise RuntimeError(
            f"Modelo {type(model).__name__} não expõe predict_proba nem "
            "decision_function; não é possível estimar probabilidade."
        )

    # ----------------------------------------------------------------- predict

    def predict_lr(self, features: np.ndarray) -> float:
        X = self._preprocess(features, self.lr)
        return self._positive_proba(self.lr, X)

    def predict_rf(self, features: np.ndarray) -> float:
        X = self._preprocess(features, self.rf)
        return self._positive_proba(self.rf, X)

    def predict_svm(self, features: np.ndarray) -> float:
        X = self._preprocess(features, self.svm)
        return self._positive_proba(self.svm, X)

    def predict_all(self, features: np.ndarray) -> dict:
        lr = self.predict_lr(features)
        rf = self.predict_rf(features)
        svm = self.predict_svm(features)
        return {
            "lr": lr,
            "rf": rf,
            "svm": svm,
            "mean": (lr + rf + svm) / 3.0,
            "feature_order": list(self.feature_order),
        }
