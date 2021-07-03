from fpdf import FPDF
import warnings
import datetime
class GeneratePDF():
    def __init__(self,pdfSavePath='', imagePath='', paramValue = [], className = [], classValue = [],page=[]):
        super(GeneratePDF, self).__init__()
        self.param = ['测试人员','试验时间','试样名称','试样编号','试样来源','委托单号','视场数量','结果文件']
        self.paramValue = paramValue
        self.className = className
        self.classValue = classValue
        self.page = page
        self.imagePath = imagePath

        # self.pdf = FPDF(format='letter',unit='cm')
        self.pdf = FPDF(format='A4',unit='cm')
        # self.pdf.add_font('fireflysung', '', 'Core/Font/fireflysung.ttf', uni=True)
        self.pdf.add_font('fireflysung', '', 'Font/fireflysung.ttf', uni=True)
        self.pdf.add_page()
        self.pdf.set_left_margin(2.18)
        self.pdf.set_top_margin(2.54)
        self.epw = self.pdf.w - 2 * self.pdf.l_margin
        self.savePath = pdfSavePath
        warnings.filterwarnings("ignore")
        try:
            self.WritePdf()
        except Exception as e:
            print(e)
#
#
    def WritePdf(self):

        self.pdf.ln(0.75)

        self.col_width = self.epw / 3
        self.pdf.set_font('fireflysung', '', 12)
        # 页眉
        self.pdf.cell(self.col_width, 0, "宝钢研究院环境与资源研究所")
        self.pdf.cell(self.col_width, 0, "")
        self.pdf.cell(self.col_width, 0, "报告编号：")


        self.pdf.ln(1)
        self.pdf.set_font('fireflysung', '', 18)
        self.col_width = self.epw / 6
        self.pdf.line(self.pdf.l_margin, 2, self.pdf.w - self.pdf.l_margin, 2)
        self.pdf.cell(self.epw, 0.0, u'矿相分析试验报告', align='C')
        self.pdf.set_font('fireflysung', '', 14)
        self.pdf.ln(0.5)
        self.pdf.cell(self.epw, 0.0, u'试验参数:', align='L')
        self.pdf.ln(1.0)

        self.pdf.set_font('fireflysung', '', 12)
        for i in range(1, 9):
            if i % 2 == 1:
                self.pdf.cell(self.col_width / 4, 0, ' ')
            self.pdf.cell(self.col_width, 0, self.param[i - 1] + ':')
            self.pdf.cell(self.col_width * 2, 0, self.paramValue[i - 1])
            if i % 2 == 0:
                self.pdf.ln(0.8)

        self.pdf.set_fill_color(128, 128, 128)
        self.pdf.set_font('fireflysung', '', 14)
        self.pdf.ln(0.5)
        self.pdf.cell(self.epw, 0.0, u'试验结果:', align='L')
        self.pdf.ln(1.0)
        self.pdf.set_font('fireflysung', '', 12)
        self.pdf.cell(self.col_width / 4, 0, '')
        self.pdf.cell(self.col_width * 3, 1, '成份', fill=True)
        self.pdf.cell(self.col_width * 2, 1, '含量', fill=True)
        self.pdf.ln(1.0)

        for i in range(0, 7):
            self.pdf.cell(self.col_width / 4, 0, '')
            self.pdf.cell(self.col_width * 3, 1, self.className[i])
            self.pdf.cell(self.col_width * 3, 1, str(self.classValue[i]))
            self.pdf.ln(0.7)
        self.pdf.ln(0.3)
        # paint image
        # self.pdf.cell(self.col_width * 1.5, 0, '')
        # self.pdf.image(self.imagePath, w=18, h=9)
        self.pdf.image(self.imagePath, w=self.epw, h=self.epw//2)
        self.pdf.ln(1.05)

        # 页脚
        self.pdf.cell(self.epw, 0.0, u'第{0}页/共{1}页'.format(self.page[0], self.page[1]), align='C')

        self.pdf.output(self.savePath, 'F')

# paramValue = ['xxx','2011年01月20日 10：00-13：00','xx','xxx','xx','自动分析方式','399','dssdfs.txt']
# className = ['矿1','矿2','矿3','矿4','矿5','矿6','矿7']
# classValue = [10,10,10,10,10,10,10]
# page = [20,50]
# writePDf = GeneratePDF('a5.pdf','../h.png',param,paramValue,className,classValue,page)
#

# writePDf.WritePdf('D:/collect2.png',param,paramValue,className,classValue,page)

