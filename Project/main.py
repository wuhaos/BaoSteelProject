from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from UI import mainWindow, snapSetting,snapSetting_1

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    aw = snapSetting_1.Ui_Form()
    w = QtWidgets.QMainWindow()
    aw.setupUi(w)
    w.show()
    sys.exit(app.exec_())


