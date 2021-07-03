from PyQt5 import QtWidgets
from PyQt5 import QtChart
from PyQt5 import QtGui
from PyQt5.Qt import Qt
from matplotlib import pyplot as plt

import sys

class ReportImage(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(900,450)

    def generatePie(self,className = [],classSum = [],imagePath=''):
        series = QtChart.QPieSeries()
        for i in range(0,len(classSum)):
            series.append(className[i] + ' ' +str(classSum[i]) + '%', classSum[i])
        font = QtGui.QFont("宋体", 12)
        font.setBold(True)
        for i in range(0,len(classSum)):
            slice0 = series.slices()[i]
            slice0.setLabelVisible(True)
            slice0.setLabelFont(font)

        chart = QtChart.QChart()
        chart.setAnimationOptions(QtChart.QChart.NoAnimation)
        chart.legend().hide()
        chart.addSeries(series)
        # chart.createDefaultAxes()

        chartview = QtChart.QChartView(chart)
        self.setCentralWidget(chartview)
        chartview.setRenderHint(QtGui.QPainter.Antialiasing)
        img = chartview.grab()
        img.save(imagePath,"PNG")
        pass

    def generateHistogram(self,className = [],classSum = [],imagePath=''):
        '''
        不考虑使用
        :param className:
        :param classSum:
        :return:
        '''
        set0 = QtChart.QBarSet('X')
        set0.append(classSum)
        series = QtChart.QBarSeries()
        # series = QtChart.QAbstractBarSeries()
        series.append(set0)

        chart = QtChart.QChart()
        chart.addSeries(series)
        label = []
        for i in range(0,len(classSum)):
            tmp = className[i] + ' ' + str(classSum[i]) + '%'
            label.append(tmp)
        axisX = QtChart.QBarCategoryAxis()
        axisX.append(label)
        axisX.setLabelsAngle(75)



        font = QtGui.QFont("宋体", 12)
        font.setBold(True)
        axisX.setLabelsFont(font)
        chart.addAxis(axisX,Qt.AlignBottom)


        series.attachAxis(axisX)
        chart.setAnimationOptions(QtChart.QChart.NoAnimation)

        chartview = QtChart.QChartView(chart)
        self.setCentralWidget(chartview)
        chartview.setRenderHint(QtGui.QPainter.Antialiasing)
        img = chartview.grab()
        img.save("a.png", "PNG")

    def generateHistStep2(self,className = [],classSum = [],imagePath=''):
        # 解决中文显示问题
        plt.figure(figsize=(9,4.5))
        plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

        plt.bar(className, height=classSum)
        # xlocs, xlabs = plt.xticks()
        xlocs = [i*1.08 - 0.6 for i in range(0, 9)]
        plt.xticks(xlocs, className,rotation = 20, fontsize = 12)
        for i, v in enumerate(classSum):
            plt.text(xlocs[i],v+0.01, str(classSum[i]) + '%',fontsize = 12)
        plt.savefig(imagePath)
        plt.show()



App = QtWidgets.QApplication(sys.argv)
window = ReportImage()
# window.generatePie(['溶蚀型磁铁矿','针状铁酸钙','柱状铁酸钙','玻璃箱和溶剂','原生赤铁矿','二次赤铁矿','字型Σ(っ °Д °;)磁铁矿','孔洞','其它'],[0.89,0.11,0.11,0.11,0.12,0.23,0.50,0.45,0.99])
# window.generateHistogram(['溶蚀型磁铁矿','针状铁酸钙','柱状铁酸钙','玻璃箱和溶剂','原生赤铁矿','二次赤铁矿','字型Σ(っ °Д °;)磁铁矿','孔洞','其它'],[9000,1000,4000,6000,40,60,5000,23000,14000],['0.89%','0.11%','0.11%','0.11%','0.11%','0.11%','0.11%','0.11%','0.11%'])
window.generateHistStep2(['溶蚀型磁铁矿','针状铁酸钙','柱状铁酸钙','玻璃箱和溶剂','原生赤铁矿','二次赤铁矿','字型磁铁矿','孔洞','其它'],[0.89,0.11,0.11,0.11,0.12,0.23,0.50,0.45,0.99])
window.show()
sys.exit(App.exec_())
