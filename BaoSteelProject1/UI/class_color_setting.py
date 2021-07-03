# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QPushButton, QColorDialog, QWidget, QLabel
from PyQt5.QtGui import QColor


class ColorSelect(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 600, 900)
        self.color = [QColor(128, 0, 0,128), QColor(255, 0, 0,255), QColor(0, 0, 0,80), QColor(0, 0, 0,60), QColor(0, 0, 0,50),
                      QColor(0, 0, 0,40), QColor(0, 0, 0,30), QColor(0, 0, 0,20), QColor(0, 0, 0,-1)]
        self.label = [QLabel(self), QLabel(self), QLabel(self), QLabel(self), QLabel(self), QLabel(self), QLabel(self),
                      QLabel(self), QLabel(self)]
        self.button = [QPushButton(self), QPushButton(self), QPushButton(self), QPushButton(self), QPushButton(self),
                       QPushButton(self), QPushButton(self), QPushButton(self), QPushButton(self)]
        self.button[0].clicked.connect(lambda: self.modifyColor(0))
        self.button[1].clicked.connect(lambda: self.modifyColor(1))
        self.button[2].clicked.connect(lambda: self.modifyColor(2))
        self.button[3].clicked.connect(lambda: self.modifyColor(3))
        self.button[4].clicked.connect(lambda: self.modifyColor(4))
        self.button[5].clicked.connect(lambda: self.modifyColor(5))
        self.button[6].clicked.connect(lambda: self.modifyColor(6))
        self.button[7].clicked.connect(lambda: self.modifyColor(7))
        self.button[8].clicked.connect(lambda: self.modifyColor(8))
        start = 50
        for idx in range(9):
            self.label[idx].setGeometry(20, start, 131, 21)
            self.button[idx].setGeometry(180, start, 131, 21)
            start += 60
            self.label[idx].setVisible(False)
            self.button[idx].setVisible(False)
            self.button[idx].setEnabled(False)

    def modifyColor(self, i):
        col = QColorDialog.getColor()
        if col.isValid():
            self.color[i] = col
            self.button[i].setStyleSheet('QPushButton{background-color:%s} ' % col.name())


# from PyQt5 import QtWidgets
# import sys
#
#
# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     mainWindow = ColorSelect()
#     mainWindow.show()
#     sys.exit(app.exec_())
#

