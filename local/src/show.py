# -*- encoding: UTF-8 -*-
import matplotlib.pyplot as plt
import cv2
import glob
import re
from mpl_toolkits.mplot3d import Axes3D

def show_images(img_path):
    """
    画像を表示する
    Args :
        img_path (string) : 画像のpath
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    plt.close()

def show_skeleton(d):
    """
    3D空間上の座標データをAxes3Dを使ってplotする
    Args :
        d (numpy.array) : 3D空間上の座標データ
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    pairs = [[6,7],[7,8],[8,9],[9,10],[10,11],
             [9,3],[8,2],[2,1],[1,0],[3,4],[4,5],
             [12,9],[12,8],[12,13],[2,3]]
    
    for i,pair in enumerate(pairs):
        x = [d[pair[0]][0], d[pair[1]][0]]
        y = [-d[pair[0]][1], -d[pair[1]][1]]
        z = [d[pair[0]][2], d[pair[1]][2]]

        if i in [0,1]:
            c = "#1f77b4" # blue
        elif i in [2,11,12,13]:
            c = "#ff7f0e" # orange
        elif i in [3,4]:
            c = "#2ca02c" # green
        elif i in [6,7,8]:
            c = "#d62728" # red
        elif i in [5,9,10]:
            c = "#9467bd" # purple
        else:
            c = "#17becf" # sky blue
        ax.plot(x,z,y,"o-", color=c)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")

    # リアルタイム
    plt.show()
    # 画像保存
    for angle in range(0, 360):
        ax.view_init(elev=15, azim=angle)
        if angle % 10 == 0:
            fig.savefig("pictures/fig_" + str(angle) + ".png")
    filepath_list = glob.glob("pictures/*.png")
    filepath_list = sorted(filepath_list, key=lambda x:int((re.search(r"[0-9]+", x)).group(0)))