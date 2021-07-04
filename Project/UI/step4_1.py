from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui


class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent)
        self.setGeometry(300, 300, 680, 800)
        # self.setFixedSize(2400,1800)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.pixmapItem = (
            QtWidgets.QGraphicsPixmapItem()
        )  # check if everytime you open a new image the old image is still an item
        self.scene().addItem(self.pixmapItem)

        self._path_item = None
        self.img = None
        self.filename = ""

    def initial_path(self):
        self._path = QtGui.QPainterPath()
        pen = QtGui.QPen(
            QtGui.QColor("green"), 4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap
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
            self._path.lineTo(self.mapToScene(event.pos()))
            self._path_item.setPath(self._path)
        super(GraphicsView, self).mouseMoveEvent(event)

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



class MainWindow(QWidget):
    def __init__(self,parent = None):
        super(MainWindow, self).__init__(parent)
        self.setGeometry(0,0,1800,1900)
        self.view = GraphicsView()
        self.selectImage = QtWidgets.QPushButton(self)
        self.selectImage.setGeometry(QtCore.QRect(20, 30, 101, 41))
        self.selectImage.setText("选择图片")
        # self.selectImage.setObjectName("selectImage")
        self.analyse = QtWidgets.QPushButton(self)
        self.analyse.setGeometry(QtCore.QRect(190, 30, 101, 41))
        self.analyse.setText("分析")
        # self.analyse.setObjectName("analyse")
        self.selectImage.clicked.connect(self.view.setImage)
        self.analyse.clicked.connect(self.view.save)
        # self.setLayout(QVBoxLayout())
        self.h1_layout = QHBoxLayout()
        self.h2_layout = QHBoxLayout()
        self.h1_layout.addWidget(self.selectImage)
        self.h1_layout.addWidget(self.analyse)
        # self.layout().addWidget(self.selectImage)
        # self.layout().addWidget(self.analyse)
        self.h2_layout.addWidget(self.view)
        self.v_layout = QVBoxLayout()
        self.v_layout.addLayout(self.h1_layout)
        self.v_layout.addLayout(self.h2_layout)
        self.setLayout(self.v_layout)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    # w = QWidget()
    # btnSave = QPushButton("Save image")
    # view = GraphicsView()
    # view.setImage()
    # w.setLayout(QVBoxLayout())
    # w.layout().addWidget(btnSave)
    # w.layout().addWidget(view)
    # btnSave.clicked.connect(lambda: view.save())
    w.show()
    sys.exit(app.exec_())

