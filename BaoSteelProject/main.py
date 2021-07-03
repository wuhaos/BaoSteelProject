from UI import bigmainwindow
from PyQt5 import QtWidgets
import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = bigmainwindow.BigMainWindow(mainWindow)
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
