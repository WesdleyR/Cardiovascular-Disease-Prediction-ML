"""
Front-end PySide6 para o preditor de risco cardiovascular.

A janela carrega os modelos em background, mostra um splash com
progresso, e em seguida exibe o formulário e o painel de resultados.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from predictor import Predictor
from utils import montar_vetor_features, validar_dados
from worker import PredictionController


# Diretório padrão de modelos (pasta `Modelos/` ao lado deste arquivo).
DEFAULT_MODELOS_DIR = Path(__file__).resolve().parent / "Modelos"


# ---------------------------------------------------------------------------
# Loader one-shot (carrega Predictor num QThread).
# ---------------------------------------------------------------------------

class _ModelLoader(QObject):
    finished = Signal(object)   # Predictor
    failed = Signal(str)

    def __init__(self, modelos_dir: Path) -> None:
        super().__init__()
        self._dir = modelos_dir

    @Slot()
    def run(self) -> None:
        try:
            predictor = Predictor(self._dir)
            self.finished.emit(predictor)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Janela principal.
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Predição de Risco Cardiovascular")
        self.resize(560, 760)

        self.predictor: Predictor | None = None
        self.controller: PredictionController | None = None

        # Estado transitório: пока грузим модели — показываем splash.
        self._loader_thread: QThread | None = None
        self._loader: _ModelLoader | None = None

        self._build_loading_splash()

        # Inicia carregamento dos modelos em background.
        self._start_model_loading()

    # --------------------------------------------------------------- loading

    def _build_loading_splash(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setAlignment(Qt.AlignCenter)

        self._loading_label = QLabel("Carregando modelos…", central)
        font = QFont()
        font.setPointSize(14)
        self._loading_label.setFont(font)
        self._loading_label.setAlignment(Qt.AlignCenter)

        self._loading_progress = QProgressBar(central)
        self._loading_progress.setRange(0, 0)  # indeterminado
        self._loading_progress.setTextVisible(False)

        layout.addStretch(1)
        layout.addWidget(self._loading_label)
        layout.addWidget(self._loading_progress)
        layout.addStretch(1)

        self.setCentralWidget(central)

    def _start_model_loading(self) -> None:
        self._loader_thread = QThread(self)
        self._loader = _ModelLoader(DEFAULT_MODELOS_DIR)
        self._loader.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader.run)
        self._loader.finished.connect(self._on_models_loaded)
        self._loader.failed.connect(self._on_models_failed)
        # Limpeza
        self._loader.finished.connect(self._loader_thread.quit)
        self._loader.failed.connect(self._loader_thread.quit)
        self._loader_thread.finished.connect(self._cleanup_loader)
        self._loader_thread.start()

    @Slot(object)
    def _on_models_loaded(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self.controller = PredictionController(predictor)
        self.controller.finished.connect(self._on_prediction_finished)
        self.controller.failed.connect(self._on_prediction_failed)
        self._build_ui()

    @Slot(str)
    def _on_models_failed(self, msg: str) -> None:
        QMessageBox.critical(
            self,
            "Erro ao carregar modelos",
            f"Não foi possível carregar os modelos:\n\n{msg}",
        )
        # Mantém o splash visível; deixa o usuário fechar a janela.

    def _cleanup_loader(self) -> None:
        if self._loader is not None:
            self._loader.deleteLater()
            self._loader = None
        if self._loader_thread is not None:
            self._loader_thread.deleteLater()
            self._loader_thread = None

    # ------------------------------------------------------------------- ui

    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        outer.addWidget(self._build_form())
        outer.addLayout(self._build_button_row())

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        outer.addWidget(self._build_result_panel())
        outer.addStretch(1)

        self.setCentralWidget(central)
        self._build_menu()

    def _build_form(self) -> QWidget:
        group = QGroupBox("Dados do paciente")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        # Idade
        self.in_age = QSpinBox()
        self.in_age.setRange(1, 120)
        self.in_age.setSuffix(" anos")
        self.in_age.setValue(50)
        form.addRow("Idade:", self.in_age)

        # Gênero
        self.in_gender = QComboBox()
        self.in_gender.addItem("Feminino", 1)
        self.in_gender.addItem("Masculino", 2)
        self.in_gender.setCurrentIndex(1)
        form.addRow("Gênero:", self.in_gender)

        # Altura
        self.in_height = QSpinBox()
        self.in_height.setRange(50, 250)
        self.in_height.setSuffix(" cm")
        self.in_height.setValue(170)
        form.addRow("Altura:", self.in_height)

        # Peso
        self.in_weight = QDoubleSpinBox()
        self.in_weight.setRange(20.0, 400.0)
        self.in_weight.setSuffix(" kg")
        self.in_weight.setDecimals(1)
        self.in_weight.setSingleStep(0.5)
        self.in_weight.setValue(70.0)
        form.addRow("Peso:", self.in_weight)

        # Pressão sistólica
        self.in_ap_hi = QSpinBox()
        self.in_ap_hi.setRange(50, 250)
        self.in_ap_hi.setSuffix(" mmHg")
        self.in_ap_hi.setValue(120)
        form.addRow("Pressão sistólica:", self.in_ap_hi)

        # Pressão diastólica
        self.in_ap_lo = QSpinBox()
        self.in_ap_lo.setRange(30, 200)
        self.in_ap_lo.setSuffix(" mmHg")
        self.in_ap_lo.setValue(80)
        form.addRow("Pressão diastólica:", self.in_ap_lo)

        # Colesterol
        self.in_cholesterol = QComboBox()
        self.in_cholesterol.addItem("Normal", 1)
        self.in_cholesterol.addItem("Acima", 2)
        self.in_cholesterol.addItem("Muito acima", 3)
        form.addRow("Colesterol:", self.in_cholesterol)

        # Glicose
        self.in_gluc = QComboBox()
        self.in_gluc.addItem("Normal", 1)
        self.in_gluc.addItem("Acima", 2)
        self.in_gluc.addItem("Muito acima", 3)
        form.addRow("Glicose:", self.in_gluc)

        # Checkboxes
        self.in_smoke = QCheckBox("Fumante")
        form.addRow("", self.in_smoke)

        self.in_alco = QCheckBox("Consome álcool")
        form.addRow("", self.in_alco)

        self.in_active = QCheckBox("Atividade física")
        self.in_active.setChecked(True)
        form.addRow("", self.in_active)

        return group

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_calcular = QPushButton("Calcular risco")
        self.btn_calcular.setDefault(True)
        self.btn_calcular.clicked.connect(self._on_calcular)
        self.btn_limpar = QPushButton("Limpar")
        self.btn_limpar.clicked.connect(self._on_limpar)
        row.addWidget(self.btn_calcular)
        row.addWidget(self.btn_limpar)
        row.addStretch(1)
        return row

    def _build_result_panel(self) -> QWidget:
        group = QGroupBox("Resultado")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.out_lr = QLabel("Regressão Logística: —")
        self.out_rf = QLabel("Random Forest: —")
        self.out_svm = QLabel("SVM: —")
        for lbl in (self.out_lr, self.out_rf, self.out_svm):
            layout.addWidget(lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        self.out_risk = QLabel("Risco consolidado: —")
        risk_font = QFont()
        risk_font.setPointSize(18)
        risk_font.setBold(True)
        self.out_risk.setFont(risk_font)
        self.out_risk.setAlignment(Qt.AlignCenter)
        self.out_risk.setMinimumHeight(56)
        self.out_risk.setStyleSheet(self._risk_style("idle"))
        layout.addWidget(self.out_risk)

        self.out_band = QLabel("Faixa: —")
        self.out_band.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.out_band)

        return group

    def _build_menu(self) -> None:
        menu = QMenuBar(self)
        self.setMenuBar(menu)

        arq = menu.addMenu("&Arquivo")
        act_limpar = QAction("&Limpar", self)
        act_limpar.setShortcut("Ctrl+L")
        act_limpar.triggered.connect(self._on_limpar)
        arq.addAction(act_limpar)
        arq.addSeparator()
        act_sair = QAction("&Sair", self)
        act_sair.setShortcut("Ctrl+Q")
        act_sair.triggered.connect(self.close)
        arq.addAction(act_sair)

        ajuda = menu.addMenu("A&juda")
        act_sobre = QAction("&Sobre", self)
        act_sobre.triggered.connect(self._show_about)
        ajuda.addAction(act_sobre)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Sobre",
            "<b>Predição de Risco Cardiovascular v1.0</b><br><br>"
            "Modelos: Regressão Logística, Random Forest e SVM "
            "(treinados no dataset cardiovascular do Kaggle).<br><br>"
            "Pipeline: StandardScaler → PCA → classificador.",
        )

    # -------------------------------------------------------------- handlers

    def _collect_inputs(self) -> dict:
        return {
            "age_years": int(self.in_age.value()),
            "gender": int(self.in_gender.currentData()),
            "height": int(self.in_height.value()),
            "weight": float(self.in_weight.value()),
            "ap_hi": int(self.in_ap_hi.value()),
            "ap_lo": int(self.in_ap_lo.value()),
            "cholesterol": int(self.in_cholesterol.currentData()),
            "gluc": int(self.in_gluc.currentData()),
            "smoke": int(self.in_smoke.isChecked()),
            "alco": int(self.in_alco.isChecked()),
            "active": int(self.in_active.isChecked()),
        }

    def _on_calcular(self) -> None:
        if self.predictor is None or self.controller is None:
            QMessageBox.warning(
                self,
                "Modelos não carregados",
                "Aguarde o carregamento dos modelos antes de calcular.",
            )
            return

        try:
            dados = self._collect_inputs()
            validar_dados(dados)
        except ValueError as exc:
            QMessageBox.warning(self, "Dados inválidos", str(exc))
            return

        try:
            features = montar_vetor_features(dados, self.predictor.feature_order)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Erro ao montar features",
                f"Não foi possível montar o vetor de features:\n\n{exc}",
            )
            return

        # Desabilita o botão e mostra progresso enquanto a predição roda.
        self.btn_calcular.setEnabled(False)
        self.progress.setVisible(True)
        self.controller.start(features)

    def _on_limpar(self) -> None:
        self.in_age.setValue(50)
        self.in_gender.setCurrentIndex(1)  # Masculino
        self.in_height.setValue(170)
        self.in_weight.setValue(70.0)
        self.in_ap_hi.setValue(120)
        self.in_ap_lo.setValue(80)
        self.in_cholesterol.setCurrentIndex(0)
        self.in_gluc.setCurrentIndex(0)
        self.in_smoke.setChecked(False)
        self.in_alco.setChecked(False)
        self.in_active.setChecked(True)

        self.out_lr.setText("Regressão Logística: —")
        self.out_rf.setText("Random Forest: —")
        self.out_svm.setText("SVM: —")
        self.out_risk.setText("Risco consolidado: —")
        self.out_risk.setStyleSheet(self._risk_style("idle"))
        self.out_band.setText("Faixa: —")

    @Slot(dict)
    def _on_prediction_finished(self, result: dict) -> None:
        self.btn_calcular.setEnabled(True)
        self.progress.setVisible(False)

        def pct(p: float) -> str:
            return f"{p * 100:.1f}%"

        self.out_lr.setText(f"Regressão Logística: {pct(result['lr'])}")
        self.out_rf.setText(f"Random Forest: {pct(result['rf'])}")
        self.out_svm.setText(f"SVM: {pct(result['svm'])}")

        consolidated = float(result["mean"])
        band = self._band_for(consolidated)
        self.out_risk.setText(f"Risco consolidado: {pct(consolidated)}")
        self.out_risk.setStyleSheet(self._risk_style(band))
        self.out_band.setText(f"Faixa: {self._band_label(band)}")

    @Slot(str)
    def _on_prediction_failed(self, msg: str) -> None:
        self.btn_calcular.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Erro de predição", msg)

    # ------------------------------------------------------------- formatação

    @staticmethod
    def _band_for(p: float) -> str:
        if p < 0.20:
            return "low"
        if p < 0.50:
            return "medium"
        return "high"

    @staticmethod
    def _band_label(band: str) -> str:
        return {"low": "Baixo", "medium": "Moderado", "high": "Alto"}.get(band, "—")

    @staticmethod
    def _risk_style(band: str) -> str:
        # Reset (cinza claro) para estado "sem predição".
        if band == "idle":
            return (
                "QLabel { background-color: #eeeeee; color: #555555;"
                " border-radius: 6px; padding: 8px; }"
            )
        if band == "low":
            return (
                "QLabel { background-color: #2e7d32; color: white;"
                " border-radius: 6px; padding: 8px; }"
            )
        if band == "medium":
            return (
                "QLabel { background-color: #f9a825; color: black;"
                " border-radius: 6px; padding: 8px; }"
            )
        return (
            "QLabel { background-color: #c62828; color: white;"
            " border-radius: 6px; padding: 8px; }"
        )

    # -------------------------------------------------------------- lifecycle

    def closeEvent(self, event) -> None:  # noqa: D401
        if self.controller is not None:
            self.controller.shutdown()
        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._loader_thread.quit()
            self._loader_thread.wait(2000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
