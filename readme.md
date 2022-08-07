# Pepper imitates human poses

## Overview
Pepper imitates 3D poses from 2D images.
Pepperが撮影した画像データに映る人間のポーズを真似する．
[HybrIK](https://github.com/Jeff-sjtu/HybrIK)を利用し，画像データから3D空間上の座標データを取り出した．

## Requirement
remote(GPU Server)
- Ubuntu 20.04.4
- Python3.7
- pyenv anaconda3-2022.05

local
- Windows7
- Python2.7

## Usage
ディレクトリ構造は以下のようにする．

remote
```
root
│
├src
 　├data
   　├camera.jpg (撮影した画像)
 　├HybrIK (git clone : [HybrIK](https://github.com/Jeff-sjtu/HybrIK))
 　├process.py
 　├server.py
```
local
```
root
│
├src
 　├pictures (骨格を出力した画像が入る)
 　├calc.py
 　├client.py
 　├main.py
 　├pepper.py
 　├show.py
 　├sound.py
```

以下のように実行する．
remote
`python ./server.py`
local
`python ./main.py`

## Reference
https://github.com/Jeff-sjtu/HybrIK
https://github.com/GVLabRobotics/pepper-blazepose
https://recruit.gmo.jp/engineer/jisedai/blog/pepper-alredballdetection/
https://www.youtube.com/watch?v=55y5BVeH9vc
https://developer.softbankrobotics.com/pepper-naoqi-25/naoqi-developer-guide/sdks/python-sdk
https://qiita.com/fukasawah/items/32ebef9cd646a1eb3e3a