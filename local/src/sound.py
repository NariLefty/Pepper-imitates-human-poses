# -*- encoding: UTF-8 -*-
from playsound import playsound
import winsound
from naoqi import ALProxy

def play_camera(path="camera.wav"):
    """
    カメラのシャッター音を鳴らす
    Args :
        path (string) : wavファイルのpath
    """
    winsound.PlaySound(path, winsound.SND_FILENAME)

def play_number(text, robotIP="192.168.11.1"):
    """
    カメラを取るまでのカウントダウンをPepperに話させる
    Args :
        text (string)    : Pepperが話す内容(e.x. '1')
        robotIP (string) : PepperのIP
    """
    tts = ALProxy("ALTextToSpeech", robotIP, 9999)
    tts.say(text)

