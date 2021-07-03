from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import numpy as np
import qimage2ndarray
import cv2 as cv


class backImage(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(backImage, self).__init__(parent)
        self.wid = 1200
        self.heit = 900
        self.setFixedSize(self.wid, self.heit)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.pixmapItem = (
            QtWidgets.QGraphicsPixmapItem()
        )  # check if everytime you open a new image the old image is still an item
        self.scene().addItem(self.pixmapItem)
        self._path_item = None
        self.img = None
        self.filename = ""
        self.setFrameStyle(QFrame.NoFrame)

    def initial_path(self):
        self._path = QtGui.QPainterPath()
        pen = QtGui.QPen(
            QtGui.QColor("white"), 1.5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap
        )
        self._path_item = self.scene().addPath(self._path, pen)

    # @QtCore.pyqtSlot()
    def setImage(self, width, height):
        self.scene().clear()
        self.wid = width
        self.heit = height
        print("width: {0} height {1}".format(width, height))
        self.scene().setSceneRect(0, 0, self.wid, self.heit)
        self.pixmapItem = (
            QtWidgets.QGraphicsPixmapItem()
        )  # check if everytime you open a new image the old image is still an item
        self.scene().addItem(self.pixmapItem)
        self.setFixedSize(width, height)

    def mousePress(self, pos):
        try:
            self.initial_path()

            self._path.moveTo(self.mapToScene(pos))
            self._path_item.setPath(self._path)
            self.scene().addItem(self.pixmapItem)

            self._path_item.setPath(self._path)
        except Exception as e:
            print("press event {0}".format(e))
        # super(backImage, self).mousePressEvent(event)

    def mouseMove(self, pos):
        self._path.lineTo(self.mapToScene(pos))
        self._path_item.setPath(self._path)
        # super(backImage, self).mouseMoveEvent(event)

    def mouseRelease(self, pos):
        self._path.lineTo(self.mapToScene(pos))
        self._path.closeSubpath()
        self._path_item.setPath(self._path)
        # self._path_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
        self._path_item.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        self._path_item = None

        # super(backImage, self).mouseReleaseEvent(event)

    def fullSelect(self):
        self.initial_path()
        self._path.moveTo(self.mapToScene(QPoint(1, 1)))
        self._path_item.setPath(self._path)
        self.scene().addItem(self._path_item)
        self._path_item.setPath(self._path)

        self._path.lineTo(self.mapToScene(QPoint(self.wid - 1, 1)))
        self._path_item.setPath(self._path)

        self._path.lineTo(self.mapToScene(QPoint(self.wid - 1, self.heit - 1)))
        self._path_item.setPath(self._path)

        self._path.lineTo(self.mapToScene(QPoint(1, self.heit - 1)))
        self._path_item.setPath(self._path)

        self._path.lineTo(self.mapToScene(QPoint(1, 1)))
        self._path.closeSubpath()
        self._path_item.setPath(self._path)
        self._path_item.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        self._path_item = None

        pass

    def save(self):
        rect = self.scene().sceneRect()
        self.img = QtGui.QImage(rect.width(), rect.height(), QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(self.img)
        rectf = QRectF(0, 0, self.img.rect().width(), self.img.rect().height())
        self.scene().render(painter, rectf, rect)
        try:
            npImg = qimage2ndarray.rgb_view(self.img)
            # filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            #     None, "save Image", self.filename, "Image Files (*.png)"
            # )
            # if filename:
            #     cv.imwrite(filename,npImg)
            # cv.imshow("img",npImg)
            return cv.cvtColor(npImg,cv.COLOR_BGR2GRAY)
        except Exception as e:
            print(e)
            return None

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     w = QWidget()
#     btnSave = QPushButton("Save image")
#     view = GraphicsView()
#     view.setImage()
#     w.setLayout(QVBoxLayout())
#     w.layout().addWidget(btnSave)
#     w.layout().addWidget(view)
#     btnSave.clicked.connect(lambda: view.save())
#     w.show()
#     sys.exit(app.exec_())
