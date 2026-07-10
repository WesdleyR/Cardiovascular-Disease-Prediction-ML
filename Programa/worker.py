"""
Bridge entre o `Predictor` (CPU-bound, sem Qt) e a UI PySide6.

Roda `predict_all` num QThread para não travar a thread principal.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from predictor import Predictor


class PredictionWorker(QObject):
    """Worker que executa `Predictor.predict_all` quando recebe um `request`."""

    finished = Signal(dict)   # resultado de predict_all
    failed = Signal(str)      # mensagem de erro (pt-BR)

    def __init__(self, predictor: Predictor) -> None:
        super().__init__()
        self._predictor = predictor

    @Slot(np.ndarray)
    def run(self, features: np.ndarray) -> None:
        try:
            result = self._predictor.predict_all(features)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 — propaga qualquer erro p/ UI
            self.failed.emit(str(exc))


class PredictionController(QObject):
    """
    Dono do QThread + worker. Expõe apenas `start(features)` e os
    signals `finished` / `failed` para a UI.
    """

    request = Signal(np.ndarray)

    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, predictor: Predictor) -> None:
        super().__init__()
        self.thread = QThread()
        self.worker = PredictionWorker(predictor)
        self.worker.moveToThread(self.thread)

        # request (na thread principal) -> worker.run (na thread do worker)
        self.request.connect(self.worker.run)
        self.worker.finished.connect(self._forward_finished)
        self.worker.failed.connect(self._forward_failed)

        self.thread.start()

    def start(self, features: np.ndarray) -> None:
        # Emite via signal para que o slot rode no QThread do worker.
        self.request.emit(features)

    def shutdown(self) -> None:
        self.thread.quit()
        self.thread.wait(2000)

    @Slot(dict)
    def _forward_finished(self, result: dict) -> None:
        self.finished.emit(result)

    @Slot(str)
    def _forward_failed(self, msg: str) -> None:
        self.failed.emit(msg)
