import sys
from PyQt5 import QtWidgets, QtGui
from UI import report_setting, snap_setting, class_color_setting
import configparser


class GlobalSetting(QtWidgets.QWidget):
    def __init__(self, parent=None, subParent = None, idx = 1):
        super(GlobalSetting, self).__init__(parent)
        self.configPath = "./File/config/test.conf"
        self.parent = subParent
        self.qTab = QtWidgets.QTabWidget(self)
        self.resize(600, 950)
        self.tab1 = report_setting.ReportSetting()
        self.tab2 = snap_setting.SnapSetting()
        self.tab3 = class_color_setting.ColorSelect()
        self.section = "step"+str(idx)
        self.qTab.addTab(self.tab1, "试验报告设置")
        self.qTab.addTab(self.tab2, "自动分析扫描参数")
        self.qTab.addTab(self.tab3, "类别颜色设置")
        self.qTab.resize(600,900)
        self.diagButton = QtWidgets.QDialogButtonBox(self)
        self.applyButton = QtWidgets.QPushButton("应用")
        self.cancelButton = QtWidgets.QPushButton("取消")
        self.diagButton.addButton(self.applyButton,QtWidgets.QDialogButtonBox.AcceptRole)
        self.diagButton.addButton(self.cancelButton,QtWidgets.QDialogButtonBox.RejectRole)
        self.diagButton.setGeometry(380,820,200,200)
        self.applyButton.clicked.connect(self.onAccepted)
        self.cancelButton.clicked.connect(self.onRejected)
        self.config = configparser.RawConfigParser()
        self.config.read(self.configPath,encoding="utf-8")
        self.setDefault()

    def setDefault(self):
        self.setWindowTitle("参数设置")
        self.tab1.sample_name.setText(self.config.get(self.section, "sampleName"))
        self.tab1.sample_number.setText(self.config.get(self.section, "sampleNumber"))
        self.tab1.sampleOrigin.setText(self.config.get(self.section, "sampleOrigin"))
        self.tab1.person_name.setText((self.config.get(self.section, "testPeople")))
        self.tab1.report_number.setText(self.config.get(self.section, "reportNumber"))
        self.tab1.page_number.setText(self.config.get(self.section, "currentPage"))
        self.tab1.sum_page_number.setText(self.config.get(self.section, "sumPage"))
        self.tab1.auto_report_path.setText(self.config.get(self.section,"autoAnalysisPath"))
        self.tab1.hand_report_path.setText(self.config.get(self.section,"handAnalysisPath"))
        self.tab1.department_name.setText(self.config.get(self.section,"department"))
        if self.config.get(self.section, "displayMode") == "pie":
            self.tab1.pie.setChecked(True)
            self.tab1.histogram.setChecked(False)
        else:
            self.tab1.pie.setChecked(False)
            self.tab1.histogram.setChecked(True)

        if self.config.get(self.section, "exposuremode") == "hand":
            self.tab2.HandExposure.setChecked(True)
            self.tab2.AutoExposure.setChecked(False)
        else:
            self.tab2.HandExposure.setChecked(False)
            self.tab2.AutoExposure.setChecked(True)

        self.tab2.ExposureSlider.setValue(int(self.config.get(self.section, "exposureValue")))
        self.tab2.ExposureType.setText(self.config.get(self.section, "exposureValue"))
        self.tab2.redType.setText(self.config.get(self.section, "whitebalance.red"))
        self.tab2.greenType.setText(self.config.get(self.section, "whitebalance.green"))
        self.tab2.blueType.setText(self.config.get(self.section, "whitebalance.blue"))
        self.tab2.colorTemperature.setValue(int(self.config.get(self.section,'colortemperature')))
        self.tab2.colorTemperatureValue.setText(self.config.get(self.section,"colortemperature"))
        self.tab2.startLeftCoordinate.setText(self.config.get(self.section, "frame.startLeft"))
        self.tab2.startTopCoordinate.setText(self.config.get(self.section, "frame.startTop"))
        self.tab2.imageWidth.setText(self.config.get(self.section, "frame.width"))
        self.tab2.imageHeight.setText(self.config.get(self.section, "frame.height"))
        self.tab2.startPositionX.setText(self.config.get(self.section, "startPositionX"))
        self.tab2.endPositionX.setText(self.config.get(self.section, "endPositionX"))
        self.tab2.stepLengthX.setText(self.config.get(self.section, "stepLengthX"))
        self.tab2.startPositionY.setText(self.config.get(self.section, "startPositionY"))
        self.tab2.endPositionY.setText(self.config.get(self.section, "endPositionY"))
        self.tab2.stepLengthY.setText(self.config.get(self.section, "stepLengthY"))
        self.tab2.imageNum.setText((self.config.get(self.section, "imageNum")))
        self.tab2.originPath.setText(self.config.get(self.section,"originImagePath"))
        self.tab2.predictPath.setText(self.config.get(self.section,"predictImagePath"))
        for i in range(0, self.parent.n_class):
            self.tab3.label[i].setText(self.config.get(self.section,"classname"+str(i)))
            col = QtGui.QColor(
                int(self.config.get(self.section,'classcolor_red'+str(i))),
                int(self.config.get(self.section,'classcolor_green'+str(i))),
                int(self.config.get(self.section,'classcolor_blue'+str(i)))
            )
            self.tab3.button[i].setStyleSheet('QPushButton{background-color:%s} ' % col.name())
            self.tab3.label[i].setVisible(True)
            self.tab3.button[i].setVisible(True)
            self.tab3.button[i].setEnabled(True)
        self.setPramter()

    def onAccepted(self):
        self.config.set(self.section,"sampleName",self.tab1.sample_name.text())
        self.config.set(self.section,"sampleNumber",self.tab1.sample_number.text())
        self.config.set(self.section,"sampleOrigin",self.tab1.sampleOrigin.text())
        self.config.set(self.section,"testPeople",self.tab1.person_name.text())
        self.config.set(self.section,"reportNumber",self.tab1.report_number.text())
        self.config.set(self.section,"currentPage",self.tab1.page_number.text())
        self.config.set(self.section,"sumPage",self.tab1.sum_page_number.text())

        if self.tab1.pie.isChecked():
            self.config.set(self.section,"displaymode","pie")
        else:
            self.config.set(self.section, "displaymode", "histogram")
        if self.tab2.HandExposure.isChecked():
            self.config.set(self.section,"exposuremode","hand")
        else:
            self.config.set(self.section, "exposureMode", "auto")
        self.config.set(self.section,"exposureValue",self.tab2.ExposureSlider.value())
        self.config.set(self.section,"whitebalance.red",self.tab2.redType.text())
        self.config.set(self.section,"whitebalance.green",self.tab2.greenType.text())
        self.config.set(self.section,"whitebalance.blue",self.tab2.blueType.text())
        self.config.set(self.section,"colortemperature",self.tab2.colorTemperatureValue.text())
        self.config.set(self.section,"frame.startLeft",self.tab2.startLeftCoordinate.text())
        self.config.set(self.section,"frame.startTop",self.tab2.startTopCoordinate.text())
        self.config.set(self.section,"frame.width",self.tab2.imageWidth.text())
        self.config.set(self.section,"frame.height",self.tab2.imageHeight.text())
        self.config.set(self.section,"startPositionX",self.tab2.startPositionX.text())
        self.config.set(self.section,"startPositionY",self.tab2.startPositionY.text())
        self.config.set(self.section,"endPositionX",self.tab2.endPositionX.text())
        self.config.set(self.section,"endPositionY",self.tab2.endPositionY.text())
        self.config.set(self.section,"stepLengthX",self.tab2.stepLengthX.text())
        self.config.set(self.section,"stepLengthY",self.tab2.stepLengthY.text())
        self.config.set(self.section,"imageNum",self.tab2.imageNum.text())
        self.config.set(self.section,"autoAnalysisPath",self.tab1.auto_report_path.text())
        self.config.set(self.section,"handAnalysisPath",self.tab1.hand_report_path.text())
        self.config.set(self.section,"originImagePath",self.tab2.originPath.text())
        self.config.set(self.section,"predictImagePath",self.tab2.predictPath.text())
        self.config.set(self.section,"department",self.tab1.department_name.text())
        for i in range(0, self.parent.n_class):
            self.config.set(self.section,'classcolor_red'+str(i), str(self.tab3.button[i].palette().button().color().red()))
            self.config.set(self.section,'classcolor_green'+str(i), str(self.tab3.button[i].palette().button().color().green()))
            self.config.set(self.section,'classcolor_blue'+str(i), str(self.tab3.button[i].palette().button().color().blue()))

        cfgFile = open(self.configPath,"w",encoding="utf-8")
        self.config.write(cfgFile,space_around_delimiters=True)
        cfgFile.close()
        self.setPramter()
        self.close()

    def setPramter(self):
        self.parent.testParmter[0] = self.tab1.person_name.text()
        self.parent.testParmter[2] = self.tab1.sample_name.text()
        self.parent.testParmter[3] = self.tab1.sample_number.text()
        self.parent.testParmter[4] = self.tab1.sampleOrigin.text()
        self.parent.header[0] = self.tab1.department_name.text()
        self.parent.header[1] = self.tab1.report_number.text()
        self.parent.page[0] = self.tab1.page_number.text()
        self.parent.page[1] = self.tab1.sum_page_number.text()
        self.parent.pie = self.tab1.pie.isChecked()
        self.parent.onlineSaveReportDir = self.tab1.auto_report_path.text()
        self.parent.offlineSaveReportDir = self.tab1.hand_report_path.text()

        try:
            self.parent.startX = int(self.tab2.startPositionX.text())
        except Exception as e:
            self.parent.startX = 0
        try:
            self.parent.endX = int(self.tab2.endPositionX.text())
        except Exception as e:
            self.parent.endX = 1

        try:
            self.parent.stepLengthX = int(self.tab2.stepLengthX.text())
        except Exception as e:
            self.parent.stepLengthX = 1

        try:
            self.parent.startY = int(self.tab2.startPositionY.text())
        except Exception as e:
            self.parent.startY = 0
        try:
            self.parent.endY = int(self.tab2.endPositionY.text())
        except Exception as e:
            self.parent.endY = 1

        try:
            self.parent.stepLengthY = int(self.tab2.stepLengthY.text())
        except Exception as e:
            self.parent.stepLengthY = 1

        self.parent.isSaveImage = self.tab2.saveButton.isChecked()
        self.parent.originImageSaveDir = self.tab2.originPath.text()
        self.parent.predictImageSaveDir = self.tab2.predictPath.text()
        try:
            self.parent.frameLeft = int(self.tab2.startLeftCoordinate.text())
        except Exception as e:
            self.parent.frameLeft = 0
        try:
            self.parent.frameTop = int(self.tab2.startTopCoordinate.text())
        except Exception as e:
            self.parent.frameTop = 0
        try:
            self.parent.frameWidth = int(self.tab2.imageWidth.text())
        except Exception as e:
            self.parent.frameWidth = 200
        try:
            self.parent.frameHeight = int(self.tab2.imageHeight.text())
        except Exception as e:
            self.parent.frameHeight = 200
        try:
            self.parent.whiteBalanceRed = float(self.tab2.redType.text())
        except Exception as e:
            self.parent.whiteBalanceRed = 1.0

        try:
            self.parent.whiteBalanceGreen = float(self.tab2.greenType.text())
        except Exception as e:
            self.parent.whiteBalanceGreen = 1.0

        try:
            self.parent.whiteBalanceBlue = float(self.tab2.blueType.text())
        except Exception as e:
            self.parent.whiteBalanceBlue = 1.0

        try:
            self.parent.colorTemperature = self.tab2.colorTemperature.value()
        except Exception as e:
            self.parent.colorTemperature = 50
        self.parent.autoExposure = self.tab2.AutoExposure.isChecked()

        for i in range(0, self.parent.n_class):
            self.parent.className[i] = self.tab3.label[i].text()
            self.parent.classColor[i][0] = self.tab3.button[i].palette().button().color().blue()
            self.parent.classColor[i][1] = self.tab3.button[i].palette().button().color().green()
            self.parent.classColor[i][2] = self.tab3.button[i].palette().button().color().red()


    def onRejected(self):
        self.close()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    demo = GlobalSetting(idx=4)
    demo.show()
    sys.exit(app.exec_())
