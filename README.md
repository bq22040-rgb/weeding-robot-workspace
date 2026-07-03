# Weeding Robot Workspace

## 概要

このリポジトリは、株元除草ロボットのためのROSパッケージです。  
RealSenseカメラ画像を用いて作物・雑草を認識し、雑草の3次元位置を推定したうえで、ロボットアームへ除草動作の指令を送ることを目的としています。

主な処理の流れは以下の通りです。

1. カメラ画像から作物・雑草を認識
2. 雑草領域から代表位置を推定
3. カメラ座標系から `base_link` 座標系へ変換
4. 複数カメラの認識結果を統合
5. 雑草位置をもとにロボットアームへ動作指令を送信
6. グリッパで雑草を把持・除去

---

## 主な機能

- RealSenseカメラを用いた画像認識
- 作物・雑草・背景の認識
- 雑草の3次元位置推定
- TFによる座標変換
- 俯瞰カメラ・側面カメラ・第3カメラの結果統合
- Hungarian法による複数候補の対応付け
- Dynamixelを用いたYaw・Pitch・Y軸・グリッパ制御
- Z軸制御ノードへの指令送信
- 除草完了通知のPublish

---

## リポジトリ構成

```text
weeding-robot-workspace/
├── launch/
│   ├── crop_recognition2.launch
│   ├── crop_recognition3.launch
│   ├── crop_recognition4.launch
│   ├── crop_recognition5.launch
│   └── crop_recognition6.launch
│
├── src/
│   ├── top_recognition_node.py
│   ├── side_recognition_node.py
│   ├── third_recognition_node.py
│   ├── weeding_recognition_node.py
│   ├── weeding_recognition2_node.py
│   ├── weeding_recognition3_node.py
│   ├── weeding_recognition_node_3cam_hungarian.py
│   ├── dmxdrv.py
│   ├── weed_marker_node.py
│   └── network/
│
├── CMakeLists.txt
└── package.xml
