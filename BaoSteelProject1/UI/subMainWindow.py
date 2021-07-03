# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow2_3.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

from UI import step2_menu, step3_menu


class SubMainWindow(QtWidgets.QWidget):
    def __init__(self,parent = None, subParent = None):
        self.parent = subParent
        super(SubMainWindow, self).__init__(parent)
        self.setObjectName("Form")
        self.setBaseSize(543, 337)
        self.step2 = QtWidgets.QPushButton(self)
        self.step2.setGeometry(QtCore.QRect(80, 110, 111, 71))
        self.step2.setObjectName("step2")
        self.step3 = QtWidgets.QPushButton(self)
        self.step3.setGeometry(QtCore.QRect(280, 110, 111, 71))
        self.step3.setObjectName("step3")
        self.setWindowTitle("second")



        self.step2.clicked.connect(self.showStep2Menu)
        self.step3.clicked.connect(self.showStep3Menu)


        self.retranslateUi()


    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.step2.setText(_translate("Form", "矿物组成分析"))
        self.step3.setText(_translate("Form", "矿物尺寸分析"))

    def showStep2Menu(self):
        QtWidgets.QMessageBox.warning(self, '提示', '请确认物镜为 20X 物镜')
        self.step2Menu = step2_menu.Step2Menu(subParent=self)
        self.hide()
        self.step2Menu.show()
        self.step2Menu.showGlobalSetting()

    def showStep3Menu(self):
        QtWidgets.QMessageBox.warning(self, '提示', '请确认物镜为 20X 物镜')
        # try:
        self.step3Menu = step3_menu.Step3Menu(subParent=self)
        self.hide()
        self.step3Menu.show()
        self.step3Menu.showGlobalSetting()
        # except Exception as e:
        #     print(e)

    def closeEvent(self, event) -> None:
        event.accept()
        self.parent.setVisible(True)