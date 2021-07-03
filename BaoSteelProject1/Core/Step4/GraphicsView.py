from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui


class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent)
        self.wid = 1100
        self.heit = 600
        self.setFixedSize(self.wid,self.heit)
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
            QtGui.QColor("red"), 1.5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap
        )
        self._path_item = self.scene().addPath(self._path, pen)

    @QtCore.pyqtSlot()
    def setImage(self):
        self.filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "select Image", "", "Image Files (*.png *.jpg *jpg *.bmp)"
        )
        if self.filename:
            pixmap1 = QtGui.QPixmap(self.filename)
            self.pixmapItem.setPixmap(pixmap1)

    def mousePressEvent(self, event):
        start = event.pos()
        if (
                not self.pixmapItem.pixmap().isNull()
                and event.buttons() & QtCore.Qt.LeftButton
        ):
            self.initial_path()
            self._path.moveTo(self.mapToScene(start))
            self._path_item.setPath(self._path)
        super(GraphicsView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
                not self.pixmapItem.pixmap().isNull()
                and event.buttons() & QtCore.Qt.LeftButton
                and self._path_item is not None
        ):

            x = event.x()
            if x < 1:
                x = 1
            if x > self.wid-1:
                x = self.wid-1
            y=event.y()
            if y<1:
                y=1
            if y > self.heit-1:
                y = self.heit-1
            pos = QPoint(x,y)
            self._path.lineTo(self.mapToScene(pos))
            self._path_item.setPath(self._path)
        super(GraphicsView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
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
            self._path_item.setBrush(QtGui.QBrush(QtGui.QColor("red")))
            # self._path_item.setFlag(
            #     QtWidgets.QGraphicsItem.ItemIsSelectable, True
            # )
            self._path_item = None

        super(GraphicsView, self).mouseReleaseEvent(event)

    def save(self):
        rect = self.scene().sceneRect()
        self.img = QtGui.QImage(rect.width(), rect.height(), QtGui.QImage.Format_RGB666)
        painter = QtGui.QPainter(self.img)
        rectf = QRectF(0, 0, self.img.rect().width(), self.img.rect().height())
        self.scene().render(painter, rectf, rect)
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,"save Image", self.filename,"Image Files (*.png)"
        )
        if filename:
            self.img.save(filename)
