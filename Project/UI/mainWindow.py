# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tmpMainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(871, 754)
        self.snap_setting = QtWidgets.QPushButton(Form)
        self.snap_setting.setGeometry(QtCore.QRect(30, 90, 101, 71))
        self.snap_setting.setObjectName("snap_setting")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 200, 101, 81))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 310, 101, 81))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(260, 80, 271, 71))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(Form)
        self.pushButton_5.setGeometry(QtCore.QRect(260, 190, 271, 71))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(260, 300, 271, 71))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(Form)
        self.pushButton_7.setGeometry(QtCore.QRect(260, 430, 271, 71))
        self.pushButton_7.setObjectName("pushButton_7")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.snap_setting.setText(_translate("Form", "拍照设置"))
        self.pushButton_2.setText(_translate("Form", "开始拍照"))
        self.pushButton_3.setText(_translate("Form", "开始拍照并分析"))
        self.pushButton_4.setText(_translate("Form", "自动分析烧结矿制粒后结构特征"))
        self.pushButton_5.setText(_translate("Form", "自动分析烧结矿主要矿物组成的面积百分比"))
        self.pushButton_6.setText(_translate("Form", "自动分析矿相尺寸及尺寸分布"))
        self.pushButton_7.setText(_translate("Form", "自动分析烧结矿孔隙率及孔径分布"))
