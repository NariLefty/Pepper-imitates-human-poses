# -*- encoding: UTF-8 -*-
import socket
import pickle
import cv2
import struct
import time
import sys
from sound import play_camera, play_number


#Ref https://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data
#**********************
def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    print("n")
    print(n)
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
#**********************

def get_info(robotIP="192.168.11.1"):
    """
    server.pyに画像を送り，3D空間上の座標データを受け取る
    Args:
        robotIP (string) : PepperのIP
    Return:
        d (list) : 画像から取得した3D空間上の座標データ
    
    """
    for i in [5,4,3,2,1]:
        print(i)
        play_number(text="{}".format(i), robotIP=robotIP)
    
    cap_cam = cv2.VideoCapture(0)
    print(cap_cam.isOpened())

    ret, frame = cap_cam.read()
    if ret:
        play_camera("camera.wav")
    # print(ret)
    cv2.imwrite("./picture.jpg", frame)
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", 12345))

    #frame = cv2.imread("./picture7.jpg") #保存した画像を使用したいときpathを指定する
    print("image shape {}".format(frame.shape))
    msg = pickle.dumps(frame)
    send_msg(s, msg)

    msg = recv_msg(s)
    d = pickle.loads(msg)
    s.close()
    return d
