import sys
import os

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from interface import MainWindow


def main():
    app = QApplication(sys.argv)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    app.setWindowIcon(QIcon(os.path.join(base_dir, "heart.ico")))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()