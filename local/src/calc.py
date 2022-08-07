# -*- encoding: UTF-8 -*-
import time
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D


#Ref https://stackoverflow.com/questions/62345076/how-to-convert-a-rodrigues-vector-to-a-rotation-matrix-without-opencv-using-pyth
def rodrigues_vec_to_rotation_mat(rodrigues_vec, theta):
    """
    Convert a Rodrigues Vector to a Rotation Matrix
    """
    #theta = np.linalg.norm(rodrigues_vec)
    if theta < sys.float_info.epsilon:
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rodrigues_vec
        I = np.eye(3, dtype=float)
        r_rT = np.array([
            [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
        ])
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat

def recalc_d(d):
    """
    画像から取得した3D空間上の座標データをロドリゲスの回転公式を用いて回転させ
    地面に対して直立した3D空間上の座標を返す
    Args:
        d (list) : 画像から取得した3D空間上の座標データ
    Returns:
        d (numpy.array) : 再計算した3D空間上の座標データ
    """

    d = np.array(d)

    # 真ん中の点
    # 右肩,左肩，右腰，左腰
    mid_p = (d[8] + d[9] + d[2] + d[3]) * 0.25
    
    # 足の点
    # 右足首,左足首
    foot_p = (d[0] + d[5]) * 0.5

    v = mid_p - foot_p
    
    #法線
    n = np.array([0,-1,0])
    
    cos_t = np.dot(n,v)/(np.linalg.norm(n)*np.linalg.norm(v))

    theta = np.arccos(cos_t)

    kaitenziku = np.cross(v,n)/np.linalg.norm(np.cross(v,n))

    mat = rodrigues_vec_to_rotation_mat(kaitenziku, theta)

    d = np.dot(mat, d.T).T
    
    return d

