import socket
import pickle
import struct
import cv2
from process import get_joints


def send_msg(sock, msg):
    print("message")
    print(len(msg))
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
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("localhost", 12345))  # IPとポート番号を指定
s.listen(5)

clientsocket, address = s.accept()
print(f"Connection from {address} has been established!")

msg = recv_msg(clientsocket)
d = pickle.loads(msg, encoding="latin1")
cv2.imwrite("./data/camera.jpg", d)

joints = get_joints()
#print(joints)

msg = pickle.dumps(joints, protocol=2) # ローカルはPython2.7であるためprotocolを指定している
send_msg(clientsocket, msg)

clientsocket.close()