from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import numpy as np
# from .step4_backImage import backImage
from UI import step4_backImage

class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent)
        self.wid = 1100
        self.heit = 600
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
        self.backimg = step4_backImage.backImage()
        # self.backimg = step4_backImage.backImage(parent=self)
        self.loadImage = False

    def initial_path(self):
        self._path = QtGui.QPainterPath()
        pen = QtGui.QPen(
            QtGui.QColor("red"), 1.5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap
        )
        self._path_item = self.scene().addPath(self._path, pen)

    # @QtCore.pyqtSlot()
    def setImage(self,fileName):
    # def setImage(self):
        self.scene().clear()
        # self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(
        #     None, "select Image", "", "Image Files (*.png *.jpg *jpg *.bmp)"
        # )
        self.filename = fileName
        if self.filename:
            self.loadImage = True
            pixmap1 = QtGui.QPixmap(self.filename)

            self.wid = pixmap1.size().width() // 2
            self.heit = pixmap1.size().height() // 2
            pixmap1 = pixmap1.scaled(self.wid, self.heit, Qt.IgnoreAspectRatio)
            self.scene().setSceneRect(0, 0, self.wid, self.heit)
            self.pixmapItem = (
                QtWidgets.QGraphicsPixmapItem()
            )  # check if everytime you open a new image the old image is still an item
            self.scene().addItem(self.pixmapItem)
            self.setFixedSize(pixmap1.size())
            self.pixmapItem.setPixmap(pixmap1)
            try:
                self.backimg.setImage(self.wid, self.heit)
            except Exception as e:
                print(e)

    def mousePressEvent(self, event):
        if self.loadImage == False:
            return
        start = event.pos()
        if (
                not self.pixmapItem.pixmap().isNull()
                and event.buttons() & QtCore.Qt.LeftButton
        ):

            self.initial_path()
            self._path.moveTo(self.mapToScene(start))
            self._path_item.setPath(self._path)
            self.scene().addItem(self.pixmapItem)
            self._path_item.setPath(self._path)
            try:
                self.backimg.mousePress(start)
            except Exception as e:
                print("press {0}".format(e))
        super(GraphicsView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.loadImage == False:
            return
        if (
                not self.pixmapItem.pixmap().isNull()
                and event.buttons() & QtCore.Qt.LeftButton
                and self._path_item is not None
        ):

            x = event.x()
            if x < 1:
                x = 1
            if x > self.wid - 1:
                x = self.wid - 1
            y = event.y()
            if y < 1:
                y = 1
            if y > self.heit - 1:
                y = self.heit - 1
            pos = QPoint(x, y)
            self._path.lineTo(self.mapToScene(pos))
            self._path_item.setPath(self._path)
            try:
                self.backimg.mouseMove(pos)
            except Exception as e:
                print("move {0}".format(e))
        super(GraphicsView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.loadImage == False:
            return
        x = event.x()
        if x < 1:
            x = 1
        if x > self.wid - 1:
            x = self.wid - 1
        y = event.y()
        if y < 1:
            y = 1
        if y > self.heit - 1:
            y = self.heit - 1
        end = QPoint(x, y)
        if (
                not self.pixmapItem.pixmap().isNull()
                and self._path_item is not None
        ):
            self._path.lineTo(self.mapToScene(end))

            self._path.closeSubpath()
            self._path_item.setPath(self._path)
            # self._path_item.setBrush(QtGui.QBrush(QtGui.QColor("red")))
            # self._path_item.setBrush(QtGui.QBrush(QtGui.QColor(255,0,0,128)))

            self._path_item = None
        try:
            self.backimg.mouseRelease(end)
        except Exception as e:
            print("release {0}".format(e))
        super(GraphicsView, self).mouseReleaseEvent(event)

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
        # self._path_item.setBrush(QtGui.QBrush(QtGui.QColor("red")))
        # self._path_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 128)))
        self._path_item = None
        try:
            self.backimg.fullSelect()
        except Exception as e:
            print(e)

        pass

    def save(self):
        try:
            return self.backimg.save()
        except Exception as e:
            print(e)
        # rect = self.scene().sceneRect()
        # # self.img = QtGui.QImage(rect.width(), rect.height(), QtGui.QImage.Format_RGB666)
        # self.img = QtGui.QImage(rect.width(), rect.height(), QtGui.QImage.Format_ARGB32)
        # painter = QtGui.QPainter(self.img)
        # rectf = QRectF(0, 0, self.img.rect().width(), self.img.rect().height())
        # self.scene().render(painter, rectf, rect)
        # filename, _ = QtWidgets.QFileDialog.getSaveFileName(
        #     None, "save Image", self.filename, "Image Files (*.png)"
        # )
        # if filename:
        #     self.img.save(filename)

        # self.img = self.img.convertToFormat(4)
        # ptr = self.img.constBits()
        # ptr.setsize(self.img.byteCount())
        # mat = np.array(ptr).reshape(self.img.height(),self.img.width(),4)
        # return mat


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = QWidget()
    btnSave = QPushButton("Save image")
    view = GraphicsView()
    view.setImage()
    w.setLayout(QVBoxLayout())
    w.layout().addWidget(btnSave)
    w.layout().addWidget(view)
    btnSave.clicked.connect(lambda: view.save())
    w.show()
    sys.exit(app.exec_())
