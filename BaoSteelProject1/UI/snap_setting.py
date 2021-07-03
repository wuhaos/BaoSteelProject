import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtWidgets,QtGui

class SnapSetting(QWidget):
    def __init__(self, parent=None):
        super(SnapSetting, self).__init__(parent)

        self.setGeometry(100, 100, 600, 900)
        self.Exposure = QGroupBox("AxioCam 506 color: 曝光时间", self)
        self.WhiteBalance = QGroupBox("AxioCam 506 color: 白平衡", self)
        self.FrameSize = QGroupBox("AxioCam 506 color: 视域", self)
        self.ImageSave = QGroupBox("保存路径", self)
        self.XAxisRange = QGroupBox("x轴范围",self)
        self.YAxisRange = QGroupBox("y轴范围",self)

        self.Exposure.setGeometry(10, 10, 400, 130)
        self.WhiteBalance.setGeometry(10, 150, 400, 130)
        self.FrameSize.setGeometry(10, 290, 400, 130)
        self.ImageSave.setGeometry(10, 430, 400, 130)
        self.XAxisRange.setGeometry(10,570,400,130)
        self.YAxisRange.setGeometry(10,710,400,130)
        # 曝光
        self.HandExposure = QRadioButton("手动曝光", self)
        self.AutoExposure = QRadioButton("拍摄时自动曝光", self)
        self.ExposureType = QLineEdit()
        self.ExposureUnit = QLabel("ms")
        self.ExposureSlider = QSlider(Qt.Horizontal)
        self.h11_layout = QHBoxLayout()
        self.h12_layout = QHBoxLayout()
        self.v11_layout = QVBoxLayout()
        self.ExposureLayout_Init()
        #     白平衡
        self.RedLabel = QLabel("红色")
        self.GreenLabel = QLabel("绿色")
        self.BlueLabel = QLabel("蓝色")
        self.colorLabel = QLabel("色温")
        self.colorTemperatureValue = QLabel("色温")
        self.redType = QLineEdit()
        self.greenType = QLineEdit()
        self.blueType = QLineEdit()
        self.colorTemperature = QSlider(Qt.Horizontal)
        self.h21_layout = QHBoxLayout()
        self.h22_layout = QHBoxLayout()
        self.h23_layout = QHBoxLayout()
        self.h24_layout = QHBoxLayout()
        self.v21_layout = QVBoxLayout()
        self.WhiteBalanceLayout_Init()
        #     视域
        self.position = QLabel("起始坐标")
        self.startLeftCoordinate = QLineEdit()
        self.interval1 = QLabel(" / ")
        self.startTopCoordinate = QLineEdit()
        self.imageSize = QLabel("图像尺寸")
        self.imageWidth = QLineEdit()
        self.interval2 = QLabel(" x ")
        self.imageHeight = QLineEdit()
        self.h31_layout = QHBoxLayout()
        self.h32_layout = QHBoxLayout()
        self.v31_layout = QVBoxLayout()
        self.FrameSizeLayout_Init()

        self.saveButton = QRadioButton(self.ImageSave)
        self.saveButton.setText("保存图片")
        self.originLabel = QLabel("原图保存路径")
        self.originPath = QLineEdit("C:/")
        self.originPathSelect = QPushButton("重新选择")
        self.predictLabel = QLabel("预测图保存路径")
        self.predictPath = QLineEdit("C:/")
        self.predictPathSelect = QPushButton("重新选择")
        self.h41_layout = QHBoxLayout()
        self.h42_layout = QHBoxLayout()
        self.h43_layout = QHBoxLayout()
        self.v41_layout = QVBoxLayout()
        self.FileSaveLayout_Init()
        self.cancelButton = QPushButton("取消")
        self.cancelButton.setGeometry(10,900,70,40)
        self.okButton = QPushButton("确认")
        self.okButton.setGeometry(50, 900, 70, 40)

        self.label_7 = QtWidgets.QLabel(self.XAxisRange)
        self.label_7.setGeometry(QtCore.QRect(30, 80, 51, 16))
        self.label_9 = QtWidgets.QLabel(self.YAxisRange)
        self.label_9.setGeometry(QtCore.QRect(30, 80, 51, 16))
        self.label_12 = QtWidgets.QLabel(self.YAxisRange)
        self.label_12.setGeometry(QtCore.QRect(30, 30, 51, 16))
        self.label_13 = QtWidgets.QLabel(self.YAxisRange)
        self.label_13.setGeometry(QtCore.QRect(150, 30, 21, 16))
        self.label_14 = QtWidgets.QLabel(self)
        self.label_14.setGeometry(QtCore.QRect(30, 848, 71, 16))
        self.label_15 = QtWidgets.QLabel(self)
        self.label_15.setGeometry(QtCore.QRect(160, 848, 51, 16))
        self.startPositionX = QtWidgets.QLineEdit(self.XAxisRange)
        self.startPositionX.setGeometry(QtCore.QRect(90, 30, 51, 20))
        self.label_10 = QtWidgets.QLabel(self.XAxisRange)
        self.label_10.setGeometry(QtCore.QRect(30, 30, 51, 16))
        self.label1 = QtWidgets.QLabel(self.XAxisRange)
        self.label1.setGeometry(QtCore.QRect(150, 30, 21, 16))
        self.endPositionX = QtWidgets.QLineEdit(self.XAxisRange)
        self.endPositionX.setGeometry(QtCore.QRect(180, 30, 51, 20))
        self.stepLengthX = QtWidgets.QLineEdit(self.XAxisRange)
        self.stepLengthX.setGeometry(QtCore.QRect(90, 80, 51, 20))
        self.imageNum = QtWidgets.QLabel(self)
        self.imageNum.setGeometry(QtCore.QRect(110, 850, 54, 12))
        self.startPositionY = QtWidgets.QLineEdit(self.YAxisRange)
        self.startPositionY.setGeometry(QtCore.QRect(90, 30, 51, 20))
        self.endPositionY = QtWidgets.QLineEdit(self.YAxisRange)
        self.endPositionY.setGeometry(QtCore.QRect(180, 30, 51, 20))
        self.stepLengthY = QtWidgets.QLineEdit(self.YAxisRange)
        self.stepLengthY.setGeometry(QtCore.QRect(90, 80, 51, 20))
        self.startPositionX.textChanged.connect(self.getResultNum)
        self.endPositionX.textChanged.connect(self.getResultNum)
        self.startPositionY.textChanged.connect(self.getResultNum)
        self.endPositionY.textChanged.connect(self.getResultNum)
        self.stepLengthX.textChanged.connect(self.getResultNum)
        self.stepLengthY.textChanged.connect(self.getResultNum)
        self.retranslateUi()

    def retranslateUi(self):
        self.ExposureType.setValidator(QtGui.QDoubleValidator(0.01, 1000, 2))
        self.redType.setValidator(QtGui.QDoubleValidator(0.01,65535.0,2))
        self.greenType.setValidator(QtGui.QDoubleValidator(0.01,65535.0,2))
        self.blueType.setValidator(QtGui.QDoubleValidator(0.01,65535.0,2))
        self.startLeftCoordinate.setValidator(QtGui.QIntValidator(0,2900))
        self.startTopCoordinate.setValidator(QtGui.QIntValidator(0,2900))
        self.imageWidth.setValidator(QtGui.QIntValidator(0,2900))
        self.imageHeight.setValidator(QtGui.QIntValidator(0,2900))
        self.startPositionY.setValidator(QtGui.QIntValidator(0,65535))
        self.startPositionX.setValidator(QtGui.QIntValidator(0,65535))
        self.endPositionY.setValidator(QtGui.QIntValidator(0,65535))
        self.endPositionX.setValidator(QtGui.QIntValidator(0,65535))
        self.stepLengthX.setValidator(QtGui.QIntValidator(0,65535))
        self.stepLengthY.setValidator(QtGui.QIntValidator(0,65535))
        _translate = QtCore.QCoreApplication.translate
        self.label_15.setText(_translate("Form", "辐结果"))
        self.label_14.setText(_translate("Form", "总计要输出"))
        self.YAxisRange.setTitle(_translate("Form", "y轴范围"))
        self.label_9.setText(_translate("Form", "步长"))
        self.startPositionY.setText(_translate("Form", "0"))
        self.label_12.setText(_translate("Form", "范围"))
        self.label_13.setText(_translate("Form", "……"))
        self.endPositionY.setText(_translate("Form", "1000"))
        self.stepLengthY.setText(_translate("Form", "100"))
        self.XAxisRange.setTitle(_translate("Form", "x轴范围"))
        self.label_7.setText(_translate("Form", "步长"))
        self.startPositionX.setText(_translate("Form", "0"))
        self.label_10.setText(_translate("Form", "范围"))
        self.label1.setText(_translate("Form", "……"))
        self.endPositionX.setText(_translate("Form", "1000"))
        self.stepLengthX.setText(_translate("Form", "100"))
        self.imageNum.setText(_translate("Form", "100"))
        # self.page_number.setValidator(QtGui.QIntValidator(1, 65535))
        # self.sum_page_number.setValidator(QtGui.QIntValidator(1, 65535))
        # self.ExposureType.setValidator(QtGui.QDoubleValidator(0.1,1000.0,2))
        # self.redType.setValidator(QtGui.QDoubleValidator(0.01,3.0,2))



    def ExposureLayout_Init(self):
        self.ExposureSlider.setMinimum(0.25)
        self.ExposureSlider.setMaximum(1000)
        self.ExposureSlider.setSingleStep(1.0)
        self.ExposureSlider.setTickInterval(1.0)
        self.ExposureSlider.setTickPosition(QSlider.TicksAbove)
        self.ExposureSlider.valueChanged.connect(self.sliderToText)
        self.ExposureType.textChanged.connect(self.textToSlider)
        self.HandExposure.setChecked(False)
        self.AutoExposure.setChecked(True)
        self.ExposureType.setEnabled(False)
        self.ExposureSlider.setEnabled(False)
        self.HandExposure.toggled.connect(self.radioButtonChanged)

        self.h11_layout.addWidget(self.ExposureType)
        self.h11_layout.addWidget(self.ExposureUnit)
        self.h11_layout.addWidget(self.ExposureSlider)
        self.h12_layout.addWidget(self.HandExposure)
        self.h12_layout.addWidget(self.AutoExposure)
        self.v11_layout.addLayout(self.h12_layout)
        self.v11_layout.addLayout(self.h11_layout)
        self.Exposure.setLayout(self.v11_layout)

    def WhiteBalanceLayout_Init(self):
        self.colorTemperature.setMinimum(0)
        self.colorTemperature.setMaximum(100)
        self.colorTemperature.setSingleStep(1)
        self.colorTemperature.setTickInterval(5)
        self.colorTemperature.setTickPosition(QSlider.TicksAbove)
        self.colorTemperature.valueChanged.connect(self.colorTemperatureChanged)

        self.h21_layout.addWidget(self.RedLabel)
        self.h21_layout.addWidget(self.redType)
        self.h22_layout.addWidget(self.GreenLabel)
        self.h22_layout.addWidget(self.greenType)
        self.h23_layout.addWidget(self.BlueLabel)
        self.h23_layout.addWidget(self.blueType)
        self.h24_layout.addWidget(self.colorLabel)
        self.h24_layout.addWidget(self.colorTemperature)
        self.h24_layout.addWidget(self.colorTemperatureValue)
        self.v21_layout.addLayout(self.h21_layout)
        self.v21_layout.addLayout(self.h22_layout)
        self.v21_layout.addLayout(self.h23_layout)
        self.v21_layout.addLayout(self.h24_layout)
        self.WhiteBalance.setLayout(self.v21_layout)

    def FrameSizeLayout_Init(self):
        self.h31_layout.addWidget(self.position)
        self.h31_layout.addWidget(self.startLeftCoordinate)
        self.h31_layout.addWidget(self.interval1)
        self.h31_layout.addWidget(self.startTopCoordinate)
        self.h32_layout.addWidget(self.imageSize)
        self.h32_layout.addWidget(self.imageWidth)
        self.h32_layout.addWidget(self.interval2)
        self.h32_layout.addWidget(self.imageHeight)
        self.v31_layout.addLayout(self.h31_layout)
        self.v31_layout.addLayout(self.h32_layout)
        self.FrameSize.setLayout(self.v31_layout)

    def FileSaveLayout_Init(self):
        self.originPathSelect.clicked.connect(self.selectOriginPath)
        self.predictPathSelect.clicked.connect(self.selectPredictPath)
        self.saveButton.toggled.connect(self.saveOriginPredictImage)
        self.h43_layout.addWidget(self.saveButton)
        self.h41_layout.addWidget(self.originLabel)
        self.h41_layout.addWidget(self.originPath)
        self.h41_layout.addWidget(self.originPathSelect)
        self.h42_layout.addWidget(self.predictLabel)
        self.h42_layout.addWidget(self.predictPath)
        self.h42_layout.addWidget(self.predictPathSelect)
        self.v41_layout.addLayout(self.h43_layout)
        self.v41_layout.addLayout(self.h41_layout)
        self.v41_layout.addLayout(self.h42_layout)
        self.ImageSave.setLayout(self.v41_layout)
        self.originPath.setEnabled(False)
        self.predictPath.setEnabled(False)

    def sliderToText(self):
        self.ExposureType.setText(str(self.ExposureSlider.value()))

    def textToSlider(self):
        if self.ExposureType.text() == "":
            return
        try:
            if self.ExposureType.text()[-1] == '。':
                self.ExposureType.setText(self.ExposureType.text()[:-1]+'.')
            self.ExposureSlider.setValue(int(self.ExposureType.text().split('.')[0]))
        except Exception as e:
            print(e)

    def radioButtonChanged(self):
        if self.HandExposure.isChecked():
            self.ExposureType.setEnabled(True)
            self.ExposureSlider.setEnabled(True)
        else:
            self.ExposureType.setEnabled(False)
            self.ExposureSlider.setEnabled(False)

    def colorTemperatureChanged(self):
        self.colorTemperatureValue.setText(str(self.colorTemperature.value()))

    def selectOriginPath(self):
        dir = QFileDialog.getExistingDirectory()
        self.originPath.setText(dir)
        return

    def selectPredictPath(self):
        dir = QFileDialog.getExistingDirectory()
        self.predictPath.setText(dir)
        return

    def getResultNum(self):
        try:
            startX = int(self.startPositionX.text())
        except Exception as e:
            startX = 0
            # self.startPositionX.setText("0")
        try:
            endX = int(self.endPositionX.text())
        except Exception as e:
            endX = 0
            # self.endPositionX.setText("0")
        try:
            stepX = int(self.stepLengthX.text())
        except Exception as e:
            stepX = 1
            # self.stepLengthX.setText("1")
        try:
            startY = int(self.startPositionY.text())
        except Exception as e:
            startY = 0
            # self.startPositionY.setText("0")
        try:
            endY = int(self.endPositionY.text())
        except Exception as e:
            endY = 0
            # self.endPositionY.setText("0")
        try:
            stepY = int(self.stepLengthY.text())
        except Exception as e:
            stepY = 0
            # self.stepLengthY.setText("1")
        try:
            numX = (endX - startX + stepX - 1) // stepX
            numY = (endY - startY + stepY - 1) // stepY
            numImage = numY * numX
        except Exception as e:
            numImage = 0
        self.imageNum.setText(str(numImage))

    def saveOriginPredictImage(self):
        if self.saveButton.isChecked():
            self.originPathSelect.setEnabled(True)
            self.predictPathSelect.setEnabled(True)
        else:
            self.originPathSelect.setEnabled(False)
            self.predictPathSelect.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 是PyQt的整个后台管理的命脉
    form = SnapSetting()  # 调用MainWindow类，并进行显示
    form.show()
    sys.exit(app.exec_())  # 运行主循环，必须调用此函数才可以开始事件处理

