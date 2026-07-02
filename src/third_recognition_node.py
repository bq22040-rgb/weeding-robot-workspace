#!/usr/bin/env python3
import os
import sys
import cv2
import rospy
import rospkg
import torch
import torch.nn as nn
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
from torchvision import transforms as T
from PIL import Image as PILImage
from tqdm import tqdm

try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False
    print("[ThirdRecognition] pyrealsense2 not found. Running without RealSense intrinsics.")

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('crop_recognition2')
sys.path.append(os.path.join(pkg_path, 'src'))

import network


class ThirdRecognitionNode:
    def __init__(self):
        rospy.init_node('third_recognition', anonymous=True)

        # セグメンテーションモデル
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

        self.bridge = CvBridge()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.target_class = 2
        self.run_once = False

        # ==========================
        # ACT用マスクpublish設定
        # ==========================
        self.act_mask_enable = rospy.get_param("~act_mask_enable", False)
        self.act_mask_rate = rospy.get_param("~act_mask_rate", 2.0)
        self.last_act_mask_time = rospy.Time(0)

        self.act_mask_pub = rospy.Publisher(
            "/act/third_mask/image_raw",
            Image,
            queue_size=1
        )

        # ----- RealSense キャリブレーション（トップカメラ用） -----
        self.depth_intrin = None

        if RS_AVAILABLE:
            try:
                camera_info = rospy.wait_for_message(
                    '/camera3/aligned_depth_to_color/camera_info',
                    CameraInfo
                )

                self.depth_intrin = rs.intrinsics()
                self.depth_intrin.width = camera_info.width
                self.depth_intrin.height = camera_info.height
                self.depth_intrin.ppx = camera_info.K[2]
                self.depth_intrin.ppy = camera_info.K[5]
                self.depth_intrin.fx = camera_info.K[0]
                self.depth_intrin.fy = camera_info.K[4]
                self.depth_intrin.model = rs.distortion.none
                self.depth_intrin.coeffs = (
                    list(camera_info.D) if camera_info.D else [0, 0, 0, 0, 0]
                )

                rospy.loginfo(
                    "ThirdRecognition: intrinsics set from /camera3/aligned_depth_to_color/camera_info"
                )

            except Exception as e:
                self.depth_intrin = None
                rospy.logwarn(f"ThirdRecognition: failed to get CameraInfo: {e}")
        else:
            rospy.logwarn("ThirdRecognition: pyrealsense2 not available. Skip intrinsics setup.")

        # ----- 画像・深度サブスクライブ -----
        self.image_sub = rospy.Subscriber(
            '/camera3/color/image_raw',
            Image,
            self.image_callback,
            queue_size=1
        )

        self.depth_image = None
        self.depth_sub = rospy.Subscriber(
            '/camera3/aligned_depth_to_color/image_raw',
            Image,
            self.depth_callback,
            queue_size=1
        )

        self.cmd_sub = rospy.Subscriber(
            '/recognition_command',
            Float64MultiArray,
            self.command_callback,
            queue_size=1
        )

        # ----- Publisher -----
        self.result_pub = rospy.Publisher(
            '/third_recognition/result',
            Float64MultiArray,
            queue_size=10
        )

        self.seg_pub = rospy.Publisher(
            '/camera3/segmented_image',
            Image,
            queue_size=10
        )

        rospy.loginfo("third.recognition node started")
        rospy.loginfo(
            f"ThirdRecognition ACT mask enable: {self.act_mask_enable}, "
            f"rate: {self.act_mask_rate} Hz"
        )

    def load_model(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('crop_recognition2')
        model_path = os.path.join(
            package_path,
            'models',
            'lettuce_top_5000.pth'
        )

        model = network.modeling.__dict__['deeplabv3plus_resnet50'](
            num_classes=4,
            output_stride=16
        )

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(self.device)
            model.eval()
            rospy.loginfo("ThirdRecognition: Model loaded successfully")
        else:
            rospy.logerr(f"ThirdRecognition: Model checkpoint not found at {model_path}")

        return model

    def depth_callback(self, ros_depth):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                ros_depth,
                desired_encoding='passthrough'
            )
            self.depth_image = depth.astype(np.float32)

        except CvBridgeError as e:
            rospy.logerr(f"ThirdRecognition: depth CvBridge error: {e}")

    def command_callback(self, msg):
        if len(msg.data) < 2:
            return

        self.target_class = int(msg.data[0])
        start_flag = msg.data[1]

        if start_flag == 1.0:
            self.run_once = True
            rospy.loginfo(f"ThirdRecognition: start for class {self.target_class}")

    def image_callback(self, ros_image):
        now = rospy.Time.now()

        # ==========================
        # ACT用マスクを出すかどうか
        # ==========================
        do_act_mask = False

        if self.act_mask_enable:
            elapsed = (now - self.last_act_mask_time).to_sec()

            if elapsed >= 1.0 / max(self.act_mask_rate, 0.1):
                # 購読者がいるときだけDeepLab推論する
                if self.act_mask_pub.get_num_connections() > 0:
                    do_act_mask = True
                    self.last_act_mask_time = now

        # ==========================
        # 通常の認識をするかどうか
        # ==========================
        do_recognition = self.run_once

        if not do_act_mask and not do_recognition:
            return

        if do_recognition:
            self.run_once = False

        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                ros_image,
                desired_encoding='bgr8'
            )
        except CvBridgeError as e:
            rospy.logerr(f"ThirdRecognition: CvBridge error: {e}")
            return

        try:
            pred = self.predict(cv_image)

            # ==========================
            # ACT用マスクpublish
            # ==========================
            if do_act_mask:
                mask_img = self.decode_mask_only(pred)
                mask_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding="bgr8")
                mask_msg.header = ros_image.header
                self.act_mask_pub.publish(mask_msg)

            # ACT用マスクだけなら終了
            if not do_recognition:
                return

            # ==========================
            # 既存の通常認識処理
            # ==========================
            seg = self.decode_target(pred, cv_image.copy())

            pixel_positions, world_positions = self.detect_world_coordinates(seg, pred)

            for (px, py) in pixel_positions:
                cv2.circle(seg, (px, py), 5, (0, 255, 255), -1)

            seg_msg = self.bridge.cv2_to_imgmsg(seg, encoding="bgr8")
            self.seg_pub.publish(seg_msg)

            result = Float64MultiArray()

            if world_positions:
                data_to_send = []

                for pos in world_positions:
                    data_to_send.extend([
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2])
                    ])

                result.data = data_to_send
                self.result_pub.publish(result)
                rospy.loginfo(f"ThirdRecognition: Published {len(world_positions)} weeds.")

            elif pixel_positions:
                px, py = pixel_positions[0]
                result.data = [float(px), float(py)]
                self.result_pub.publish(result)
                rospy.logwarn("ThirdRecognition: depth not available, published pixel coords only")

            else:
                rospy.loginfo("ThirdRecognition: no object detected")

        except Exception as e:
            rospy.logerr(f"ThirdRecognition: error in image_callback: {e}")

    def predict(self, cv_image):
        pil_image = PILImage.fromarray(
            cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        )

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(input_tensor).max(1)[1].cpu().numpy()[0]

        return pred

    # ============================================================
    # 小さい領域を削除
    # ============================================================
    def remove_small_components(self, binary_mask, min_area):
        """
        小さい誤検出領域を削除する。
        binary_mask: 0 または 255 の2値画像
        min_area: この面積未満の領域を削除
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )

        cleaned = np.zeros_like(binary_mask)

        for label_id in range(1, num_labels):  # 0は背景
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area >= min_area:
                cleaned[labels == label_id] = 255

        return cleaned

    def decode_mask_only(self, mask):
        """
        ACT用の純粋なマスク画像を作る。

        背景 = 黒
        作物 = 緑
        雑草 = 赤

        弱めの後処理版：
        - 3x3のOPENだけ使う
        - CLOSEは使わない
        - 小さい領域を削除
        - ROI外を黒にする
        """
        h, w = mask.shape

        # クラスごとに2値化
        crop_mask = (mask == 1).astype(np.uint8) * 255
        weed_mask = (mask == 2).astype(np.uint8) * 255

        # 弱めのカーネル
        kernel_small = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        # 小さい点ノイズだけ消す：OPEN = 収縮 → 膨張
        crop_mask = cv2.morphologyEx(
            crop_mask,
            cv2.MORPH_OPEN,
            kernel_small
        )

        weed_mask = cv2.morphologyEx(
            weed_mask,
            cv2.MORPH_OPEN,
            kernel_small
        )

        # 小さい領域を削除
        crop_mask = self.remove_small_components(
            crop_mask,
            min_area=500
        )

        weed_mask = self.remove_small_components(
            weed_mask,
            min_area=150
        )

        # 重なった場合は雑草を優先
        crop_mask[weed_mask > 0] = 0

        # 色付きマスクを作る
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        color_mask[crop_mask > 0] = [0, 255, 0]  # 作物 = 緑
        color_mask[weed_mask > 0] = [0, 0, 255]  # 雑草 = 赤

        # ROI外を黒にする（全体俯瞰用：launchから変更可能）
        roi_x_min = rospy.get_param("~roi_x_min", 0)
        roi_x_max = rospy.get_param("~roi_x_max", 1280)
        roi_y_min = rospy.get_param("~roi_y_min", 0)
        roi_y_max = rospy.get_param("~roi_y_max", 720)

        roi_x_min = max(0, min(roi_x_min, w))
        roi_x_max = max(0, min(roi_x_max, w))
        roi_y_min = max(0, min(roi_y_min, h))
        roi_y_max = max(0, min(roi_y_max, h))

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[roi_y_min:roi_y_max, roi_x_min:roi_x_max] = 1

        color_mask[roi_mask == 0] = [0, 0, 0]

        return color_mask

    def decode_target(self, mask, original_image):
        """
        通常確認用のオーバーレイ画像。
        背景は元画像そのまま。
        作物 class 1 = 青
        雑草 class 2 = 赤
        """
        output = original_image.copy()

        output[mask == 1] = [255, 0, 0]
        output[mask == 2] = [0, 0, 255]

        return output

    def detect_world_coordinates(self, cv_image, pred):
        pixel_positions = []
        world_positions = []

        if (not RS_AVAILABLE) or (self.depth_intrin is None) or (self.depth_image is None):
            rospy.logwarn("ThirdRecognition: RS or intrinsics/depth not available, skip 3D deprojection.")

            original_mask = (pred == self.target_class).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            closed_mask = cv2.erode(original_mask, kernel, iterations=1)
            opened_mask = cv2.dilate(closed_mask, kernel, iterations=3)

            contours, _ = cv2.findContours(
                opened_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                M = cv2.moments(contour)

                if M["m00"] == 0:
                    continue

                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                pixel_positions.append((cX, cY))
                rospy.loginfo(f"ThirdRecognition: pixel ({cX}, {cY})")

            return pixel_positions, world_positions

        original_mask = (pred == self.target_class).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        closed_mask = cv2.erode(original_mask, kernel, iterations=1)
        opened_mask = cv2.dilate(closed_mask, kernel, iterations=3)

        contours, _ = cv2.findContours(
            opened_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ROI範囲（全体俯瞰用：launchから変更可能）
        roi_x_min = rospy.get_param("~roi_x_min", 0)
        roi_x_max = rospy.get_param("~roi_x_max", 1280)
        roi_y_min = rospy.get_param("~roi_y_min", 0)
        roi_y_max = rospy.get_param("~roi_y_max", 720)

        cv2.rectangle(
            cv_image,
            (roi_x_min, roi_y_min),
            (roi_x_max, roi_y_max),
            (0, 255, 255),
            2
        )

        for contour in contours:
            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if not (roi_x_min <= cX <= roi_x_max and roi_y_min <= cY <= roi_y_max):
                continue

            pixel_positions.append((cX, cY))

            r = 35
            h, w = self.depth_image.shape[:2]

            y1, y2 = max(0, cY - r), min(h, cY + r)
            x1, x2 = max(0, cX - r), min(w, cX + r)

            roi_depth = self.depth_image[y1:y2, x1:x2]
            valid = roi_depth[(np.isfinite(roi_depth)) & (roi_depth > 0)]

            if valid.size == 0:
                rospy.logwarn(
                    f"ThirdRecognition: no valid depth around ({cX},{cY}), skip 3D for this contour"
                )
                continue

            sorted_vals = np.sort(valid)
            cutoff_idx = int(len(sorted_vals) * 0.6)
            top_40 = sorted_vals[cutoff_idx:]

            depth_value = float(np.mean(top_40))

            Z_m = depth_value / 1000.0
            Z_m = Z_m * 0.87

            if Z_m <= 0:
                rospy.logwarn(f"ThirdRecognition: invalid Z ({Z_m}) at ({cX},{cY}), skip.")
                continue

            try:
                point_3d = rs.rs2_deproject_pixel_to_point(
                    self.depth_intrin,
                    [float(cX), float(cY)],
                    float(Z_m)
                )
                X_m, Y_m, Z_m = point_3d

            except Exception as e:
                rospy.logerr(f"ThirdRecognition: rs2_deproject_pixel_to_point error: {e}")
                continue

            X_mm = X_m * 1000
            Y_mm = Y_m * 1000
            Z_mm = Z_m * 1000

            rospy.loginfo(
                f"ThirdRecognition: pixel ({cX},{cY}) -> camera "
                f"(X={X_mm:.1f}mm, Y={Y_mm:.1f}mm, Z={Z_mm:.1f}mm)"
            )

            world_positions.append((X_mm, Y_mm, Z_mm))

        # 近接する重心を50mm以内で統合
        if world_positions:
            merged_positions = []
            used_indices = set()

            for i in range(len(world_positions)):
                if i in used_indices:
                    continue

                current_group = [world_positions[i]]
                used_indices.add(i)

                for j in range(i + 1, len(world_positions)):
                    if j in used_indices:
                        continue

                    dist = np.linalg.norm(
                        np.array(world_positions[i]) - np.array(world_positions[j])
                    )

                    if dist < 50.0:
                        current_group.append(world_positions[j])
                        used_indices.add(j)

                avg_pos = np.mean(current_group, axis=0)
                merged_positions.append(tuple(avg_pos))

            world_positions = merged_positions

        return pixel_positions, world_positions


if __name__ == '__main__':
    try:
        node = ThirdRecognitionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass