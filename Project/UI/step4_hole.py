# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'step4_1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1279, 893)
        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(0, 80, 1281, 811))
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setObjectName("graphicsView")
        self.selectImage = QtWidgets.QPushButton(Form)
        self.selectImage.setGeometry(QtCore.QRect(20, 30, 101, 41))
        self.selectImage.setObjectName("selectImage")
        self.analyse = QtWidgets.QPushButton(Form)
        self.analyse.setGeometry(QtCore.QRect(190, 30, 101, 41))
        self.analyse.setObjectName("analyse")

        self.graphicsView.setScene(QtWidgets.QGraphicsScene())
        self.pixmapItem = (
            QtWidgets.QGraphicsPixmapItem()
        )  # check if everytime you open a new image the old image is still an item
        self.graphicsView.scene().addItem(self.pixmapItem)

        self.selectImage.clicked.connect(self.setImage)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.selectImage.setText(_translate("Form", "选择图片"))
        self.analyse.setText(_translate("Form", "分析"))

        def initial_path(self):
            self._path = QtGui.QPainterPath()
            pen = QtGui.QPen(
                QtGui.QColor("green"), 4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap
            )
            self._path_item = self.scene().addPath(self._path, pen)

    def initial_path(self):
        self._path = QtGui.QPainterPath()
        pen = QtGui.QPen(
            QtGui.QColor("green"), 4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap
        )
        self._path_item = self.scene().addPath(self._path, pen)


    # @QtCore.pyqtSlot()
    def setImage(self):
        self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "select Image", "", "Image Files (*.png *.jpg *jpg *.bmp)"
        )
        if self.filename:
            pixmap1 = QtGui.QPixmap(self.filename)
            self.pixmapItem.setPixmap(pixmap1)

    def mousePressEvent(self, event):
        print("press!!")
        start = event.pos()
        if (
                not self.pixmapItem.pixmap().isNull()
                and event.buttons() & QtCore.Qt.LeftButton
        ):
            self.initial_path()
            self._path.moveTo(self.graphicsView.mapToScene(start))
            self._path_item.setPath(self._path)
        # super(GraphicsView, self).mousePressEvent(event)
        self.graphicsView.mousePressEvent(event)


    def mouseMoveEvent(self, event):
        print("mouse move")
        if (
                not self.pixmapItem.pixmap().isNull()
                and event.buttons() & QtCore.Qt.LeftButton
                and self._path_item is not None
        ):
            self._path.lineTo(self.graphicsView.mapToScene(event.pos()))
            self._path_item.setPath(self._path)
        # super(GraphicsView, self).mouseMoveEvent(event)
        self.graphicsView.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        end = event.pos()
        if (
                not self.pixmapItem.pixmap().isNull()
                and self._path_item is not None
        ):
            self._path.lineTo(self.mapToScene(end))
            self._path.closeSubpath()
            self._path_item.setPath(self._path)
            self._path_item.setBrush(QtGui.QBrush(QtGui.QColor("red")))
            self._path_item.setFlag(
                QtWidgets.QGraphicsItem.ItemIsSelectable, True
            )
            self._path_item = None

        # super(GraphicsView, self).mouseReleaseEvent(event)
        self.graphicsView.mouseReleaseEvent(event)

    def save(self):
        rect = self.scene().sceneRect()
        self.img = QtGui.QImage(rect.width(), rect.height(), QtGui.QImage.Format_RGB666)
        painter = QtGui.QPainter(self.img)
        rectf = QRectF(0, 0, self.img.rect().width(), self.img.rect().height())
        self.scene().render(painter, rectf, rect)
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "save Image", self.filename, "Image Files (*.png)"
        )
        if filename:
            self.img.save(filename)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    aw = Ui_Form()
    w = QtWidgets.QMainWindow()
    aw.setupUi(w)
    w.show()
    sys.exit(app.exec_())
