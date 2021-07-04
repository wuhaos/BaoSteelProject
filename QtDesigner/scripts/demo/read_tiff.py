from ctypes import cdll

from ctypes import *

def main():
    p = r'C:/Users/32373/source/repos/Dll1/x64/Debug/DLL1.dll'
    f = cdll.LoadLibrary(p)
    f.SayHello()
    a, b = 100, 10
    m = f.Add(a,b)
    print(m)
    m = f.Sub(a,b)
    print(m)
    m = f.Multiply(a,b)
    print(m)
    # t = f.returnString()
    path = b"C:/Windows/Containers/serviced/"
    # for i in range(7,15):
    #     # path += str(i).decode()
    #     path1 = path + str(i).encode(encoding="utf-8")
    #     print(path1)
    #     f.testString(path1)
    # //print(f.testString(c_char_p(path)))
    # print(f.testString(b"testDasddfgdall End!!!"))
    # f.testString(path)
    # print(f.DoubleMultiply(c_double(1.26768),c_longdouble(75668.4745)))
    print(f.DoubleMultiply(c_double(23.343),c_double(2342.3434)))


if __name__ == '__main__':
    main()
    # savePath ="start"
    # for i in range savePath:
    #     print(i)
    # print(savePath[1])