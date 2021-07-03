import clr
clr.AddReference(r"D:/QT/BaoSteelProject1/MTBApi.dll")
from ZEISS import MTB
import time
import math

class StageXY():
    def __init__(self):
        self.__connection = MTB.Api.MTBConnection()
        self.__mtbid = self.__connection.Login("en","")
        self.__root = MTB.Api.IMTBRoot(self.__connection.GetRoot(self.__mtbid))
        self.__unit = self.__root.GetComponent("MTBStageAxisX").GetPositionUnit(0)
        # range from -135000.0 to 135000.0
        self.__axisX = self.__root.GetComponent("MTBStageAxisX")
        # range from -88000.0 to 88000.0
        self.__axisY = self.__root.GetComponent("MTBStageAxisY")
        self.axisXLower = self.__axisX.GetMinPosition(self.__unit)
        self.axisXUpper = self.__axisX.GetMaxPosition(self.__unit)
        self.axisYLower = self.__axisY.GetMinPosition(self.__unit)
        self.axisYUpper = self.__axisY.GetMaxPosition(self.__unit)
        self.__stage = self.__root.GetComponent("MTBStage")

    def setPosition(self, x, y):
        """
        set stage absolute position at x and y axis
        :param x:
        :param y:
        :return: successed status
        """
        return self.__stage.SetPosition(float(x), float(y), self.__unit, MTB.Api.MTBCmdSetModes.Synchronous)

    def setPosition2(self, x, y):
        """
                set stage absolute position at x and y axis
                :param x:
                :param y:
                :return: successed status
        """
        self.__stage.SetPosition(float(x), float(y), self.__unit, MTB.Api.MTBCmdSetModes.Synchronous)
        startTime = time.time()
        curx, cury = 0, 0
        while(self.__stage.IsBusy):
            curx = self.getAxisXPosition()
            cury = self.getAxisYPosition()
        endTime = time.time()
        print("move time:  ", endTime - startTime)




    def getAxisXPosition(self):
        """
        get stage axis x position ---> unit micrometer
        :return: axis x position
        """
        return self.__axisX.GetPosition(self.__unit)

    def getAxisYPosition(self):
        """
        get stage axis y position ---> unit micrometer
        :return: axis y position
        """
        return self.__axisY.GetPosition(self.__unit)




