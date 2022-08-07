# -*- encoding: UTF-8 -*-

import os
import threading
from client import get_info
from pepper import instruct
from show import show_images, show_skeleton
from calc import recalc_d

def main():
    robotIP="192.168.11.1"

    #3D空間上の座標を取得
    d = get_info(robotIP=robotIP)
    d = recalc_d(d)

    # Axes3Dで座標のみをプロット
    t1 = threading.Thread(target=show_skeleton, args=(d,))
    t1.start()
    # Pepperに指示する
    instruct(d, robotIP=robotIP)
    t1.join()

if __name__ == '__main__':
    main()