import serial.tools.list_ports
from ctypes import cdll
import os
import cv2 as cv
import time

from ctypes import *
from Core.StageXY import StageXY
# import StageXY
import logging


class Camera():
    def __init__(self, acceleration=20.0, speed=100.0):
        # print(os.getcwd())
        # self.__path = r"C:/Users/32373/source/repos/ZeissHardwareControl/x64/Debug/ZeissHardwareControl.dll"
        # self.__path = r"ZeissControl2.dll"
        self.__path = "D:/QT/BaoSteelProject1/File/config/ZeissControl2.dll"
        self.__portName = "USB-SERIAL CH340"
        self.__Library = cdll.LoadLibrary(self.__path)
        self.port = -1
        self.stage = StageXY()
        self.__Init(acceleration, speed)

    def __Init(self, acceleration, speed):
        """
        initialize camera and z axis
        :return: error code
        """
        self.__GetPort()
        print("camera init", self.__Library.CameraInitialize())
        self.__Library.Z_AxisInit(self.port, c_float(acceleration), c_float(speed))

    def __Z_AxisInit(self, acceleration, speed):
        self.__Library.Z_AxisInit(self.port, c_float(acceleration), c_float(speed))

    def deInit(self):
        """
        deinitialize camera
        :return:
        """
        self.__Library.CameraDeinit();

    def __GetPort(self):
        portList = list(serial.tools.list_ports.comports())
        for i in range(0, len(portList)):
            if str(portList[i]).find(self.__portName) != -1:
                self.port = int(str(portList[i]).split(' ')[0][3:])
                return True
        return False

    def setFrameSize(self, left, right, top, bottom):
        """
        set camera frame size
        :param left:
        :param right:
        :param top:
        :param bottom:
        :return: error code
        """
        return self.__Library.SetCameraFrameSize(c_long(left), c_long(right), c_long(top), c_long(bottom))

    def setCameraExposure(self, exposureTime):
        """
        set camera exposure time
        :param exposureTime:
        :return: error code
        """
        return self.__Library.SetCameraExposure(c_long(exposureTime))

    def setCameraWhiteBalance(self, red, green, blue):
        """
        set camera white balance default red, green, blue = 0.95, 1.47, 0.58
        :param red:
        :param green:
        :param blue:
        :return:
        """
        return self.__Library.SetCameraWhiteBalance(c_double(red), c_double(green), c_double(blue))

    def AbsoluteRoughAutoFocus(self, stepLength, moveRange, width, height):
        """
        absolute position rough auto focus
        :param stepLength: pointer move step length ---> internal
        :param moveRange: z axis move range  from 0 - moverrange to moverange
        :param maxStepLengthZAxis: max step length
        :param width: image width
        :param height: image height
        :return: error code
        """
        return self.__Library.AbsoluteRoughAutoFocus(c_long(stepLength), c_int32(moveRange),

                                                     c_int32(width), c_int32(height))

    def AbsoluteFineAutoFocus(self, stepLength, moveRange, width, height):
        """
        absolute position rough auto focus
        :param stepLength: pointer move step length ---> internal
        :param moveRange: z axis move range  from 0 - moverrange to moverange
        :param maxStepLengthZAxis: max step length
        :param width: image width
        :param height: image height
        :return: error code
        """
        return self.__Library.AbsoluteFineAutoFocus(c_long(stepLength), c_int32(moveRange),
                                                    c_int32(width), c_int32(height))

    def RelativeRoughAutoFocus(self, stepLength, maxStepLengthZAxis, width, height):
        """
        absolute position rough auto focus
        :param stepLength: pointer move step length ---> internal
        :param maxStepLengthZAxis: max step length
        :param width: image width
        :param height: image height
        :return: error code
        """
        return self.__Library.RelativeRoughAutoFocus(c_long(stepLength), c_float(maxStepLengthZAxis),
                                                     c_int32(width), c_int32(height))

    def RelativeFineAutoFocus(self, stepLength, maxStepLengthZAxis, width, height):
        """
        absolute position rough auto focus
        :param stepLength: pointer move step length ---> internal
        :param maxStepLengthZAxis: max step length
        :param width: image width
        :param height: image height
        :return: error code
        """
        return self.__Library.RelativeFineAutoFocus(c_long(stepLength), c_float(maxStepLengthZAxis),
                                                    c_int32(width), c_int32(height))

    def AutoExposure(self):
        """
        absolute position rough auto focus
        :return: error code
        """
        return self.__Library.CameraAutoExposure_Sum()

    def WriteImage(self, imagePath):
        """
        save image to destination path
        :param imagePath: image save path  type::bytes
        :return:
        """
        return self.__Library.writeImage(imagePath)

    def GetCurrentExposure(self):
        return self.__Library.GetCurrentExposure()

    def MoveRelative(self, moveLength):
        self.__Library.ZAxisRelativeMoveWait(c_float(moveLength))

    def AutoFocus(self):
        curStepLength = 0.01
        minStepLength = 0.0005
        moveRange = 5
        while (curStepLength >= minStepLength):
            maxDiffer = 0
            curDiffer = 0
            print("curStepLength", curStepLength)
            curPosition = 0
            maxPosition = 0
            self.MoveRelative((0 - moveRange) * curStepLength)
            curPosition -= moveRange * curStepLength
            for i in range(0 - moveRange, moveRange):
                startTime = time.time()
                self.WriteImage(b"tmp")
                img = cv.imread("tmp.tif", 0)
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                curDiffer = cv.Laplacian(img, cv.CV_64F).var()
                if curDiffer > maxDiffer:
                    maxDiffer = curDiffer
                    maxPosition = curPosition
                moveStartTime = time.time()
                self.MoveRelative(curStepLength)
                moveEndTime = time.time()
                print("move time:", moveEndTime - moveStartTime)
                curPosition += curStepLength
                endTime = time.time()
                print(endTime - startTime)

            self.MoveRelative(maxPosition - curPosition)
            curStepLength /= 10.0

    def AutoFocusSum(self):
        self.__Library.AbsoluteAutoFocus_Sum()

    def BinaryAutoFocus(self):
        self.__Library.BinaryFocus_Sum()

    def Z_AxisSetSpeedAcceleration(self,acceleration, speed):
        self.__Library.Z_AxisSetSpeedAcceleration(c_float(acceleration), c_float(speed))

    def SetCameraImageWhiteBalance(self,red,green,blue):
        self.__Library.SetCameraImageWhiteBalance(c_double(red),c_double(green),c_double(blue))

    '''
    camera snap image 1800x2400 width: real length 870μm    height : real length 650μm 
    '''




if __name__ == "__main__":
    # savePath = b"D:/Bao_Steel_auto/MTB Demo Python Sources/test3/"
    #
    # logging.basicConfig(level=logging.INFO,
    #                                        filename="white_balance3.log",
    #                                        filemode="a+",
    #                                        format='%(message)s')
    # # os.mkdir(savePath)
    # os.chdir("C:/Users/32373/source/repos/ZeissControl2/ZeissControl2")
    camera = Camera(1.0,1.0)
    # camera.setFrameSize(176,2576,204,2004)
    # camera.setCameraWhiteBalance(0.4785909, 1.06483, 0.507396)
    # camera.WriteImage(b"origin2")
    # idx = 0
