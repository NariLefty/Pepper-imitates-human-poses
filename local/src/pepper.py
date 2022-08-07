# -*- encoding: UTF-8 -*-

from audioop import cross
from lib2to3.pgen2.token import DOT
import numpy as np
import math
import sys
from naoqi import ALProxy

# Ref https://github.com/GVLabRobotics/pepper-blazepose
def calc_info(d):
    """
    3D空間上の座標データからPepperの関節角度を計算し，指示を出す
    Args :
        d (numpy.array) : 3D空間上の座標データ
    """
    X1 = [d[9][0]]   # Left Shoulder X
    X2 = [d[8][0]]   # Right Shoulder X
    X3 = [d[10][0]]  # Left Elbow X
    X4 = [d[7][0]]   # Right Elbow X
    X5 = [d[11][0]]  # Left Wrist X
    X6 = [d[6][0]]   # Right Wrist X
    X7 = [d[3][0]]   # Left Hip X
    X8 = [d[2][0]]   # Right Hip X
    X9 = [d[14][0]]  # Nose X

    Y1 = [d[9][1]]   # Left Shoulder Y
    Y2 = [d[8][1]]   # Right Shoulder Y
    Y3 = [d[10][1]]  # Left Elbow Y
    Y4 = [d[7][1]]   # Right Elbow Y
    Y5 = [d[11][1]]  # Left Wrist Y
    Y6 = [d[6][1]]   # Right Wrist Y
    Y7 = [d[3][1]]   # Left Hip Y
    Y8 = [d[2][1]]   # Right Hip Y
    Y9 = [d[14][1]]  # Nose Y

    Z1 = [d[9][2]]   # Left Shoulder Z
    Z2 = [d[8][2]]   # Right Shoulder Z
    Z3 = [d[10][2]]  # Left Elbow Z
    Z4 = [d[7][2]]   # Right Elbow Z
    Z5 = [d[11][2]]  # Left Wrist Z
    Z6 = [d[6][2]]   # Right Wrist Z
    Z7 = [d[3][2]]   # Left Hip Z
    Z8 = [d[2][2]]   # Right Hip Z
    Z9 = [d[14][2]]  # Nose Z

    # Joint Angles in 2D
    xy = {'LeftShoulder': np.column_stack([X1, Y1]), 'RightShoulder': np.column_stack([X2, Y2]),
          'LeftElbow': np.column_stack([X3, Y3]), 'RightElbow': np.column_stack([X4, Y4]),
          'LeftWrist': np.column_stack([X5, Y5]), 'RightWrist': np.column_stack([X6, Y6]),
          'LeftHip': np.column_stack([X7, Y7]), 'RightHip': np.column_stack([X8, Y8])}
    
    # Arms landmarks
    landmark1 = 'LeftShoulder'
    landmark2 = 'LeftHip'
    landmark3 = 'LeftElbow'
    landmark4 = 'LeftWrist'
    landmark5 = 'RightShoulder'
    landmark6 = 'RightHip'
    landmark7 = 'RightElbow'
    landmark8 = 'RightWrist'

    # Calculate each body segment vector in 2D
    LS_LE = xy[landmark3] - xy[landmark1]  # Left Shoulder to Left Elbow
    LE_LS = xy[landmark1] - xy[landmark3]  # Left Elbow to Left Shoulder
    LW_LE = xy[landmark3] - xy[landmark4]  # Left Wrist to Left Elbow
    LS_LH = xy[landmark2] - xy[landmark1]  # Left Shoulder to Left Hip

    RS_RE = xy[landmark7] - xy[landmark5]  # Right Shoulder to Right Elbow
    RE_RS = xy[landmark5] - xy[landmark7]  # Right Elbow to Right Shoulder
    RW_RE = xy[landmark7] - xy[landmark8]  # Right Wrist to Right Elbow
    RS_RH = xy[landmark6] - xy[landmark5]  # Right Shoulder to Right Hip

    RS_LS = xy[landmark1] - xy[landmark5]  # Right Shoulder to Left Shoulder
    
    # Original Lengths of arm segments
    l1_left = np.linalg.norm(LE_LS[0, :])  # Upper left arm
    l2_left = np.linalg.norm(LW_LE[0, :])  # Lower left arm

    l1_right = np.linalg.norm(RE_RS[0, :])  # Upper right arm
    l2_right = np.linalg.norm(RW_RE[0, :])  # Lower right arm

    torso = Y8[0] - Y2[0]  # Torso original height
    
    # Robot/Human Angles
    LShoulderPitch = []
    RShoulderPitch = []
    LElbowYaw = []
    RElbowYaw = []
    RShoulderRoll = []
    LShoulderRoll = []
    RElbowRoll = []
    LElbowRoll = []
    HeadYaw = []
    HeadPitch = []
    HipRoll = []
    HipPitch = []

    # Calculate the hip roll angles
    for i in range(len(X1)):
        # Current horizontal distance between the 2 shoulders over the distance between them
        adj = (X1[i] - X2[i]) / np.linalg.norm(RS_LS[i, :])
        if adj >= 1:          # Keeping the ratio less than or equal to 1
            adj = 1
        phi = np.arccos(adj)  # Arc cos to get the angle
        if phi >= 0.5149:     # Maximum right hip roll is 29.5ﾂｰ.
            phi = 0.5149
        if Y2[i] < Y1[i]:     # If right shoulder is above the left shoulder then the direction of hip roll is reversed.
            phi = phi * -1
        if phi <= -0.5149:    # Maximum left hip roll is -29.5ﾂｰ.
            phi = -0.5149
        HipRoll.append(phi)

    # Calculate the hip pitch angles(Edited)
    HipPitch = []
    x0, y0, z0 = X2[0], Y2[0], Z2[0] #Right shoulder
    x1, y1, z1 = X8[0], Y8[0], Z8[0] # Right Hip 
    x2, y2, z2 = d[1][0], d[1][1], d[1][2] # Right Knee
    vec1 = [x0-x1, y0-y1, z0-z1]
    vec2 = [0,-1,0]
    
    absvec1 = np.linalg.norm(vec1)
    absvec2 = np.linalg.norm(vec2)
    inner = np.inner(vec1, vec2)

    cos_theta = inner/(absvec1 * absvec2)

    radian = abs(math.acos(cos_theta))
    theta = math.degrees(math.acos(cos_theta))
    if radian > 1.0385:
        radian = 1.0385
    
    xr, yr, zr = X2[0], Y2[0], Z2[0] #Right shoulder
    xl, yl, zl = X1[0], Y1[0], Z1[0] #Left  shoulder

    rv = np.array([xr, yr, zr])
    lv = np.array([xl, yl, zl])

    vlr = rv-lv
    cross = np.cross(vlr,np.array(vec2))
    dot = np.dot(cross,vec1) 

    if dot > 0:
        radian = radian
    else:
        radian = -radian
    for i in range(len(Y2)):
        
        HipPitch.append(radian)
    

    # Calculate the head yaw angles
    d = np.linalg.norm(RS_LS[0, :]) / 2  # Half of initial distance between right and left shoulder

    for i in range(len(X9)):
        if (X9[i] - X2[i]) / d >= 0.9 and (X9[i] - X2[i]) / d <= 1.1:  # Estimating the angle to be 0ﾂｰ if the nose
            hy = 0.0                                                   # X coordinate doesn't exceed 10% from each side

        elif (X9[i] - X2[i]) / d < 0.9:                                # Angle of looking to the right based on how much
            hy = ((d - (X9[i] - X2[i])) / d) * -(np.pi / 2)            # the nose is approaching the right shoulder.
            if hy <= -np.pi / 2:                                       # Maximum head yaw angle to the right is -90ﾂｰ.
                hy = -np.pi / 2

        elif (X9[i] - X2[i]) / d > 1.1:                                # Angle of looking to the right based on how much
            hy = (((X9[i] - X2[i]) - d) / d) * (np.pi / 2)             # the nose is approaching the left shoulder.
            if hy >= np.pi / 2:                                        # Maximum head yaw angle to the left is 90ﾂｰ.
                hy = np.pi / 2
        HeadYaw.append(hy)

    # Calculate the head pitch angles
    h = Y2[0] - Y9[0]
    for i in range(len(Y9)):
        if (Y2[i] - Y9[i]) / h >= 0.95 and (Y2[i] - Y9[i]) / h <= 1.0:
            hp = 0.0
        elif (Y2[i] - Y9[i]) / h < 0.95:
            hp = ((h - (Y2[i] - Y9[i])) / h) * 0.6371
            if hp >= 0.6371:
                hp = 0.6371
        elif (Y2[i] - Y9[i]) / h > 1.0:
            hp = (((Y2[i] - Y9[i]) - h) / h) * -0.7068 * 2
            if hp <= -0.7068:
                hp = -0.7068
        HeadPitch.append(hp)

    # 3D coordinates
    xyz = {'LeftShoulder': np.column_stack([X1, Y1, Z1]),
           'RightShoulder': np.column_stack([X2, Y2, Z2]),
           'LeftElbow': np.column_stack([X3, Y3, Z3]), 'RightElbow': np.column_stack([X4, Y4, Z4]),
           'LeftWrist': np.column_stack([X5, Y5, Z5]), 'RightWrist': np.column_stack([X6, Y6, Z6])}
    #'LeftShoulder': np.column_stack([X1, Y1, [0 for z in range(len(X1))]]),
    #'RightShoulder': np.column_stack([X2, Y2, [0 for z in range(len(X2))]]),
    
    # 3D vectors
    LS_LE_3D = xyz[landmark3] - xyz[landmark1]
    RS_RE_3D = xyz[landmark7] - xyz[landmark5]

    LE_LS_3D = xyz[landmark1] - xyz[landmark3]
    LW_LE_3D = xyz[landmark3] - xyz[landmark4]

    RE_RS_3D = xyz[landmark5] - xyz[landmark7]
    RW_RE_3D = xyz[landmark7] - xyz[landmark8]

    UpperArmLeft = xyz[landmark3] - xyz[landmark1]
    UpperArmRight = xyz[landmark7] - xyz[landmark5]

    ZeroXLeft = xyz[landmark3] - xyz[landmark1]
    ZeroXRight = xyz[landmark7] - xyz[landmark5]

    ZeroXLeft[:, 0] = 0
    ZeroXRight[:, 0] = 0

    UpperArmLeft[:, 1] = 0
    UpperArmRight[:, 1] = 0


    # Calculate the left shoulder roll angles
    for i in range(LS_LE_3D.shape[0]):
        temp1 = (np.dot(LS_LE_3D[i, :], ZeroXLeft[i, :])) / (np.linalg.norm(LS_LE_3D[i, :]) * np.linalg.norm(ZeroXLeft[i, :]))
        temp = np.arccos(temp1)
        if temp >= 1.56:
            temp = 1.56
        if temp < np.arccos((np.dot(LS_LE_3D[0, :], ZeroXLeft[0, :])) / (np.linalg.norm(LS_LE_3D[0, :]) * np.linalg.norm(ZeroXLeft[0, :]))):
            temp = 0.0
        LShoulderRoll.append(temp)

    # Calculate the right shoulder roll angles
    for i in range(RS_RE_3D.shape[0]):
        temp1 = (np.dot(RS_RE_3D[i, :], ZeroXRight[i, :])) / (np.linalg.norm(RS_RE_3D[i, :]) * np.linalg.norm(ZeroXRight[i, :]))
        temp = np.arccos(temp1)
        if temp >= 1.56:
            temp = -1.56
        else:
            temp = temp * (-1)
        if temp > -np.arccos((np.dot(RS_RE_3D[0, :], ZeroXRight[0, :])) / (np.linalg.norm(RS_RE_3D[0, :]) * np.linalg.norm(ZeroXRight[0, :]))):
            temp = 0.0
        RShoulderRoll.append(temp)

    # Calculate the left elbow roll angles
    for i in range(LE_LS_3D.shape[0]):
        temp1 = (np.dot(LE_LS_3D[i, :], LW_LE_3D[i, :])) / (np.linalg.norm(LE_LS_3D[i, :]) * np.linalg.norm(LW_LE_3D[i, :]))
        temp = np.arccos(temp1)
        if temp >= 1.56:
            temp = -1.56
        else:
            temp = temp * -1
        LElbowRoll.append(temp)

    # Calculate the right elbow roll angles
    for i in range(RE_RS_3D.shape[0]):
        temp1 = (np.dot(RE_RS_3D[i, :], RW_RE_3D[i, :])) / (np.linalg.norm(RE_RS_3D[i, :]) * np.linalg.norm(RW_RE_3D[i, :]))
        temp = np.arccos(temp1)
        if temp >= 1.56:
            temp = 1.56
        RElbowRoll.append(temp)

    # Calculate the left shoulder pitch & left elbow yaw angles
    for i in range(LE_LS_3D.shape[0]):
        temp1 = (np.dot(UpperArmLeft[i, :], LS_LE_3D[i, :])) / (np.linalg.norm(UpperArmLeft[i, :]) * np.linalg.norm(LS_LE_3D[i, :]))
        temp = np.arccos(temp1)
        if temp >= np.pi / 2:
            temp = np.pi / 2
        if Y1[i] > Y3[i]:
            temp = temp * -1
        LShoulderPitch.append(temp)

        if LShoulderRoll[i] <= 0.4:
            ley = -np.pi / 2
        elif Y3[i] - Y5[i] > 0.2 * l2_left:
            ley = -np.pi / 2
        elif Y3[i] - Y5[i] < 0 and -(Y3[i] - Y5[i]) > 0.2 * l2_left and LShoulderRoll[i] > 0.7:
            ley = np.pi / 2
        else:
            ley = 0.0
        LElbowYaw.append(ley)

    # Calculate the right shoulder pitch & right elbow yaw angles
    for i in range(RE_RS_3D.shape[0]):
        temp1 = (np.dot(UpperArmRight[i, :], RS_RE_3D[i, :])) / (np.linalg.norm(UpperArmRight[i, :]) * np.linalg.norm(RS_RE_3D[i, :]))
        temp = np.arccos(temp1)
        if temp >= np.pi / 2:
            temp = np.pi / 2
        if Y2[i] > Y4[i]:
            temp = temp * -1
        RShoulderPitch.append(temp)

        if RShoulderRoll[i] >= -0.4:
            rey = np.pi / 2
        elif Y4[i] - Y6[i] > 0.2 * l2_right:
            rey = np.pi / 2
        elif Y4[i] - Y6[i] < 0 and -(Y4[i] - Y6[i]) > 0.2 * l2_right and RShoulderRoll[i] < -0.7:
            rey = -np.pi / 2
        else:
            rey = 0.0
        RElbowYaw.append(rey)

    print("HeadYaw       : {}".format(HeadYaw[0]))
    print("HeadPitch     : {}".format(HeadPitch[0]))
    print("HipPitch      : {}".format(HipPitch[0]))
    print("HipRoll       : {}".format(HipRoll[0]))
    print("LShoulderPitch: {}".format(LShoulderPitch[0]))
    print("RShoulderPitch: {}".format(RShoulderPitch[0]))
    print("LShoulderRoll : {}".format(LShoulderRoll[0]))
    print("RShoulderRoll : {}".format(RShoulderRoll[0]))
    print("LElbowRoll    : {}".format(LElbowRoll[0]))
    print("RElbowRoll    : {}".format(RElbowRoll[0]))
    print("LElbowYaw     : {}".format(LElbowYaw[0]))
    print("RElbowYaw     : {}".format(RElbowYaw[0]))

    return [HeadYaw[0], HeadPitch[0], HipPitch[0], HipRoll[0], RShoulderPitch[0], RShoulderRoll[0], RElbowRoll[0], RElbowYaw[0], LShoulderPitch[0], LShoulderRoll[0], LElbowRoll[0], LElbowYaw[0]]
    
def instruct(d, robotIP="192.168.11.1", PORT=9999):
    ROOP_NUM = 10
    
    motionProxy = ALProxy("ALMotion", robotIP, PORT)

    #head_y, head_p, hip_p, hip_r, rshoul_p, rshoul_r, relbow_r, relbow_y,lshoul_p, lshoul_r, lelbow_r, lelbow_y
    joints_list = calc_info(d)

    names = ["HipPitch",
             "RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw",
             "LShoulderPitch", "LShoulderRoll", "LElbowRoll", "LElbowYaw"]

    angleLists = []
    for i in [2,4,5,6,7,8,9,10,11]:# namesに対応する値をjoints_listから取り出す
        angles = []
        for _ in range(ROOP_NUM):
            angles.append(joints_list[i])
        angleLists.append(angles)
                          
    timeLists = []
    for i in range(9):
        times = []
        for j in range(ROOP_NUM):
            times.append(j+5)
        timeLists.append(times)
    
    isAbsolute = True
    
    motionProxy.angleInterpolation(names, angleLists, timeLists, isAbsolute)

