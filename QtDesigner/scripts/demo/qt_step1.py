# import sys
# from PyQt5 import QtWidgets, uic
#
# app = QtWidgets.QApplication(sys.argv)
#
# window = uic.loadUi("../../UI/mainwindow.ui")
# window.show()
# app.exec()

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("../../UI/mainwindow.ui", self)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()