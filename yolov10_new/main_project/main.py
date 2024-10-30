from PySide6 import QtWidgets
from gui import VideoProcessingWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = VideoProcessingWindow()
    window.show()
    app.exec()
