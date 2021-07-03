# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'step4_menu.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import win32com.client as win32
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
from UI import query_report, global_setting, GraphicsView, report_setting
import cv2 as cv
import os
import numpy as np
import time
from Core import Camera
from skimage.measure import label
from Core.neuralNet import predict


class Step4Menu(QtWidgets.QMainWindow):
    def __init__(self, parent=None, subParent=None):
        self.parent = subParent
        super(Step4Menu, self).__init__(parent)

        self.setObjectName("step4")
        self.resize(1123, 874)

        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.next_image = QtWidgets.QPushButton(self.centralwidget)
        self.next_image.setGeometry(QtCore.QRect(70, 40, 81, 31))
        self.next_image.setObjectName("next_image")
        self.analysis = QtWidgets.QPushButton(self.centralwidget)
        self.analysis.setGeometry(QtCore.QRect(480, 40, 81, 31))
        self.analysis.setObjectName("analysis")
        self.full_select = QtWidgets.QPushButton(self.centralwidget)
        self.full_select.setGeometry(QtCore.QRect(270, 40, 81, 31))
        self.full_select.setObjectName("full_select")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1123, 21))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.parameter_setting = QtWidgets.QAction(self)
        self.parameter_setting.setObjectName("parameter_setting")
        self.real_time_setting = QtWidgets.QAction(self)
        self.real_time_setting.setObjectName("real_time_setting")
        self.auto_analysis = QtWidgets.QAction(self)
        self.auto_analysis.setObjectName("auto_analysis")
        self.handExit = QtWidgets.QAction(self)
        self.handExit.setObjectName("handExit")
        self.open_report = QtWidgets.QAction(self)
        self.open_report.setObjectName("open_report")
        self.modify_report = QtWidgets.QAction(self)
        self.modify_report.setObjectName("modify_report")
        self.save_report = QtWidgets.QAction(self)
        self.save_report.setObjectName("save_report")
        self.print_report = QtWidgets.QAction(self)
        self.print_report.setObjectName("print_report")
        self.query_report = QtWidgets.QAction(self)
        self.query_report.setObjectName("query_report")
        self.open_excel = QtWidgets.QAction(self)
        self.open_excel.setObjectName("open_excel")
        self.offline_analysis = QtWidgets.QAction(self)
        self.offline_analysis.setObjectName("offline_analysis")
        self.menu.addAction(self.parameter_setting)
        self.menu.addAction(self.real_time_setting)
        self.menu.addAction(self.auto_analysis)
        self.menu.addAction(self.handExit)
        self.menu_2.addAction(self.open_report)
        self.menu_2.addAction(self.modify_report)
        self.menu_2.addAction(self.save_report)
        self.menu_2.addAction(self.print_report)
        self.menu_2.addAction(self.query_report)
        self.menu_3.addAction(self.open_excel)
        self.menu_4.addAction(self.offline_analysis)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.painter = GraphicsView.GraphicsView(self)
        self.painter.setGeometry(QtCore.QRect(10, 120, 10, 10))
        self.reportSetting = report_setting.ReportSetting()

        self.queryReport = None
        self.globalSetting = None

        self.query_report.triggered.connect(self.showQueryReport)
        self.parameter_setting.triggered.connect(self.showGlobalSetting)
        self.open_report.triggered.connect(self.openFile)
        self.handExit.triggered.connect(self.handClose)
        self.open_excel.triggered.connect(self.openExcel)
        self.next_image.clicked.connect(self.nextImage)
        self.analysis.clicked.connect(self.analyseImage)
        self.full_select.clicked.connect(self.fullSelect)
        self.offline_analysis.triggered.connect(self.offlineAnalysis)
        self.modify_report.triggered.connect(self.modifyReport)
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        self.camera = Camera.Camera()
        self.analysisPrameter()

    def analysisPrameter(self):
        """载物台"""
        self.startX = 0
        self.endX = 1
        self.startY = 0
        self.endY = 1
        self.stepLengthX = 1
        self.stepLengthY = 1
        self.curPositionX = self.startX
        self.curPositionY = self.startY
        self.direction = True

        """保存路径"""
        self.isSaveImage = False
        self.originImageSaveDir = ""
        self.predictImageSaveDir = ""
        self.offlineSaveReportDir = ""
        self.onlineSaveReportDir = ""

        """报告所需参数"""
        self.className = ['矿1', '矿2', '矿3', '矿4', '矿5', '矿6', '矿7', '矿8', '矿9']
        self.pie = True
        self.page = ['', '']
        self.header = ['', '']
        self.testParmter = ['', '', '', '', '', '', '', '']
        self.startTime = ''
        self.endTime = ''

        """摄像头"""
        self.frameLeft = 176
        self.frameTop = 204
        self.frameWidth = 2400
        self.frameHeight = 1800
        self.autoExposure = False
        self.exposureTime = 30.0
        self.whiteBalanceRed = 120
        self.whiteBalanceGreen = 120
        self.whiteBalanceBlue = 120
        self.colorTemperature = 50

        """类别代表颜色"""
        self.classColor = [
            [0, 0, 0],
            [0, 0, 0]
        ]

        self.chkpth = "File/config/step4.pth"
        self.n_class = 2
        self.stepLength = 512
        self.Predict = predict.Predict(chkpth=self.chkpth, n_classes=self.n_class)
        self.classSum = [0, 0]
        self.imgPath = "a.png"
        self.imgIdx = 0
        self.img = None
        self.analysisMode = True
        self.offlinOriginImageList = []
        self.offlineOriginImageDir = ""
        self.holeAreaSum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.holeAreaStandard = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200000]
        for i in range(0, 10):
            self.holeAreaStandard[i] = int(pow((self.holeAreaStandard[i] / 2) / 1.44, 2) * 3.14)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("self", "step4"))
        self.next_image.setText(_translate("self", "撤销"))
        self.analysis.setText(_translate("self", "分析"))
        self.full_select.setText(_translate("self", "全选"))
        self.menu.setTitle(_translate("self", "操作"))
        self.menu_2.setTitle(_translate("self", "试验报告"))
        self.menu_3.setTitle(_translate("self", "试验数据"))
        self.menu_4.setTitle(_translate("self", "方式"))
        self.parameter_setting.setText(_translate("self", "参数设置"))
        self.real_time_setting.setText(_translate("self", "实时显示"))
        self.auto_analysis.setText(_translate("self", "自动分析"))
        self.handExit.setText(_translate("self", "退出"))
        self.open_report.setText(_translate("self", "打开文件"))
        self.modify_report.setText(_translate("self", "修改报告"))
        self.save_report.setText(_translate("self", "保存"))
        self.print_report.setText(_translate("self", "打印"))
        self.query_report.setText(_translate("self", "历史报告查询"))
        self.open_excel.setText(_translate("self", "打开Excel"))
        self.offline_analysis.setText(_translate("self", "离线分析"))
        self.next_image.setEnabled(False)
        self.full_select.setEnabled(False)
        self.analysis.setEnabled(False)

    def setDefault(self):
        import configparser
        path = "./File/config/test.conf"
        config = configparser.RawConfigParser()
        config.read(path, encoding="utf-8")
        self.globalSetting.tab1.sample_name.setText(config.get("step4", "sampleName"))
        self.globalSetting.tab1.sample_number.setText(config.get("step4", "sampleNumber"))
        self.globalSetting.tab1.sampleOrigin.setText(config.get("step4", "sampleOrigin"))
        self.globalSetting.tab1.person_name.setText((config.get("step4", "testPeople")))
        self.globalSetting.tab1.report_number.setText(config.get("step4", "reportNumber"))
        self.globalSetting.tab1.page_number.setText(str(config.get("step4", "currentPage")))
        self.globalSetting.tab1.sum_page_number.setText(str(config.get("step4", "sumPage")))
        if config.get("step4", "displayMode") == "pie":
            self.globalSetting.tab1.pie.setChecked(True)
            self.globalSetting.tab1.histogram.setChecked(False)
        else:
            self.globalSetting.tab1.pie.setChecked(False)
            self.globalSetting.tab1.histogram.setChecked(True)

        if config.get("step4", "exposureMode") == "hand":
            self.globalSetting.tab2.HandExposure.setChecked(True)
            self.globalSetting.tab2.AutoExposure.setChecked(False)
        else:
            self.globalSetting.tab2.HandExposure.setChecked(False)
            self.globalSetting.tab2.AutoExposure.setChecked(True)

        self.globalSetting.tab2.ExposureSlider.setValue(int(config.get("step4", "exposureValue")))
        self.globalSetting.tab2.ExposureType.setText(config.get("step4", "exposureValue"))
        self.globalSetting.tab2.redType.setText(config.get("step4", "whitebalance.red"))
        self.globalSetting.tab2.greenType.setText(config.get("step4", "whitebalance.green"))
        self.globalSetting.tab2.blueType.setText(config.get("step4", "whitebalance.blue"))
        self.globalSetting.tab2.startLeftCoordinate.setText(config.get("step4", "frame.startLeft"))
        self.globalSetting.tab2.startTopCoordinate.setText(config.get("step4", "frame.startTop"))
        self.globalSetting.tab2.imageWidth.setText(config.get("step4", "frame.width"))
        self.globalSetting.tab2.imageHeight.setText(config.get("step4", "frame.height"))
        self.globalSetting.tab2.startPositionX.setText(config.get("step4", "startPositionX"))
        self.globalSetting.tab2.endPositionX.setText(config.get("step4", "endPositionX"))
        self.globalSetting.tab2.stepLengthX.setText(config.get("step4", "stepLengthX"))
        self.globalSetting.tab2.startPositionY.setText(config.get("step4", "startPositionY"))
        self.globalSetting.tab2.endPositionY.setText(config.get("step4", "endPositionY"))
        self.globalSetting.tab2.stepLengthY.setText(config.get("step4", "stepLengthY"))
        self.globalSetting.tab2.imageNum.setText((config.get("step4", "imageNum")))

    def closeEvent(self, event):
        event.accept()
        self.parent.setVisible(True)

    def showQueryReport(self):
        self.queryReport = query_report.QueryReport()
        self.queryReport.show()

    def showGlobalSetting(self):
        try:
            self.globalSetting = global_setting.GlobalSetting(subParent=self, idx=4)
            self.globalSetting.show()
        except Exception as e:
            print(e)

    def openFile(self):
        # fileName = QtWidgets.QFileDialog.getOpenFileName()
        # print(fileName)
        try:
            os.system("taskkill /im msedge.exe /f")
        except Exception as e:
            print(e)

    def handClose(self):
        '''
        hand close current window and visit parent window
        :return:
        '''
        self.setVisible(False)
        self.parent.setVisible(True)

    def openExcel(self):
        '''
        open excel file
        :return:
        # '''

        fileName = QtWidgets.QFileDialog.getOpenFileName()
        if fileName[0] != '':
            out_file = Path.cwd() / fileName[0]
            excel = win32.gencache.EnsureDispatch('Excel.Application')
            excel.Visible = True
            excel.Workbooks.Open(out_file)
        return

    def nextImage(self):
        self.painter.setImage(fileName=self.imgPath)

    def analyseImage(self):
        '''
        analyseize image
        :return:
        '''
        img = cv.imread(self.imgPath)
        try:
            mark = self.Predict.predict(image=img, stepLength=self.stepLength)
        except Exception as e:
            print(e)
        paintMark = self.painter.save()
        # cv.imshow('img',paintMark)
        mark = cv.resize(mark, (paintMark.shape[1], paintMark.shape[0]), interpolation=0)
        if self.isSaveImage:
            colorImg = np.zeros((paintMark.shape[0], paintMark.shape[1], 3)).astype(np.uint8)
            colorImg[mark == 0] = self.classColor[0]
            colorImg[mark == 1] = self.classColor[1]
            predictImagePath = self.predictImageSaveDir + '/' + self.imgPath.split('/')[-1]
            cv.imwrite(predictImagePath, colorImg)
        ret1, ret2 = self.getAnalysisParmeter(mark, paintMark)
        self.classSum[0] += ret1[0]
        self.classSum[1] += ret1[1]
        for i in range(0, 10):
            self.holeAreaSum[i] += ret2[i]
        if self.analysisMode:
            self.OnlineGetOriginImage()
        else:
            self.OfflinGetOriginImage()

    def fullSelect(self):
        self.painter.fullSelect()

    def modifyReport(self):
        self.reportSetting.auto_report_path.setEnabled(False)
        self.reportSetting.hand_report_path.setEnabled(False)
        self.reportSetting.show()

    def getAnalysisParmeter(self, mark, paintMark):
        index = np.where(paintMark == 255)
        try:
            index1 = np.where((mark == 0) & (paintMark == 255))
        except Exception as e:
            print("step4 getAnalysisParmeter 1: {0}".format(e))
        ret1 = [len(index[0]), len(index1[0])]
        ret2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        mark[:, :] = 0
        mark[index1] = 1
        try:
            labels, num = label(mark, background=0, return_num=True, connectivity=2)
            for i in range(1, num + 1):
                holeArea = len(np.where(labels == i)[0])
                for j in range(0, 10):
                    if holeArea < self.holeAreaStandard[j]:
                        ret2[j] += holeArea
                        break
        except Exception as e:
            print("step4 getAnalysisParmeter 2: {0}".format(e))
        return ret1, ret2

    def autoAnalysis(self):
        self.next_image.setEnabled(True)
        self.full_select.setEnabled(True)
        self.analysis.setEnabled(True)
        self.curPositionX -= self.stepLengthX
        self.analysisMode = True
        self.imgIdx = 0
        self.OnlineGetOriginImage()

    def offlineAnalysis(self):
        self.offlineOriginImageDir = QtWidgets.QFileDialog.getExistingDirectory()
        if self.offlineOriginImageDir != '':
            self.next_image.setEnabled(True)
            self.full_select.setEnabled(True)
            self.analysis.setEnabled(True)
            self.analysisMode = False
            self.imgIdx = 0
            self.offlinOriginImageList = os.listdir(self.offlineOriginImageDir)
            self.OfflinGetOriginImage()

    def OnlineGetOriginImage(self):
        if self.direction and self.curPositionX >= self.endX:
            self.curPositionY += self.stepLengthY
            self.direction = False
        elif self.direction:
            self.curPositionX += self.stepLengthX
        elif self.direction == False and self.curPositionX <= self.startX:
            self.curPositionY += self.stepLengthY
            self.direction = True
        elif self.direction == False:
            self.curPositionX -= self.stepLengthX
        if self.curPositionY >= self.endY:
            self.next_image.setEnabled(False)
            self.analysis.setEnabled(False)
            self.full_select.setEnabled(False)
            pass

            QtWidgets.QMessageBox.warning(self, '提示', '已经分析完毕')
            return

        self.camera.stage.setPosition(self.curPositionX, self.curPositionY)
        self.camera.BinaryAutoFocus()
        if self.isSaveImage:
            self.imgPath = self.originImageSaveDir + '/' + str(self.imgIdx) + '.png'
            self.camera.WriteImage(imagePath=self.imgPath.encode(encoding='utf-8'))
            self.painter.setImage(fileName=self.imgPath)
            self.imgIdx += 1
        else:
            self.imgPath = 'a.png'
            self.camera.WriteImage(imagePath=self.imgPath.encode(encoding='utf-8'))
            self.painter.setImage(fileName=self.imgPath)

    def OfflinGetOriginImage(self):
        while self.imgIdx < len(self.offlinOriginImageList):
            suffix = self.offlinOriginImageList[self.imgIdx].split('.')[-1]
            if suffix == 'png' or suffix == 'bmp' or suffix == 'jpg' or suffix == 'tif':
                break
            else:
                self.imgIdx += 1
        if self.imgIdx < len(self.offlinOriginImageList):
            self.imgPath = self.offlineOriginImageDir + '/' + self.offlinOriginImageList[self.imgIdx]
            self.painter.setImage(fileName=self.imgPath)
            self.imgIdx += 1
        else:
            self.next_image.setEnabled(False)
            self.analysis.setEnabled(False)
            self.full_select.setEnabled(False)
            pass

            QtWidgets.QMessageBox.warning(self, '提示', '所选文件夹图片已经全部分析完毕')
