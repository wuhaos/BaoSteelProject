from UI import bigMainWindow
from PyQt5 import QtWidgets
import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = bigMainWindow.BigMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
