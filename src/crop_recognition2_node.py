import os
import rospkg
import rospy
import cv2
import torch
import torch.nn as nn
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
from torchvision import transforms as T
from PIL import Image as PILImage
import sys
import time
# --- tf2 関連の追加インポート ---
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import pyrealsense2 as rs
sys.path.append('/home/jetpack513/weed_ws3/src/crop_recognition2/src')
import network
from tqdm import tqdm


class CropRecognitionNode:
    def __init__(self):
        rospy.init_node('crop_recognition2_node', anonymous=True)

        # Define transformations
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        rospy.loginfo("Crop recognition2 node initialized for integrated two-camera system")

        # Default class for stalk estimation (class 1: coriander)
        self.target_class = 1

        # Set up the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

        self.bridge = CvBridge()

        # Run flag (per detection cycle)
        self.run_detection = False

        # Camera type mapping
        self.camera_types = {
            0: 'top',      # camera1 = 上からのカメラ（XY座標用）
            1: 'side'      # camera2 = 横からのカメラ（Z座標用）
        }

        # Detection results buffer per camera
        self.detection_results = {
            'top':  {'positions': [], 'timestamp': None, 'processed': False},
            'side': {'positions': [], 'timestamp': None, 'processed': False}
        }

        # Per-camera reentrancy latch (確実に1フレームだけ通す)
        self.processing_flags = {'top': False, 'side': False}

        # 座標統合用のタイムアウト（秒）
        self.integration_timeout = 2.0

        # --- Setup for two cameras (queue_size=1 to avoid backlog) ---
        # Camera 1 topics (Top camera)
        self.image_sub1 = rospy.Subscriber(
            '/camera1/color/image_raw', Image, self.image_callback, callback_args=0, queue_size=1
        )
        # Camera 2 topics (Side camera)
        self.image_sub2 = rospy.Subscriber(
            '/camera2/color/image_raw', Image, self.image_callback, callback_args=1, queue_size=1
        )

        # Depth topics for depth images (optional, keep queue small)
        self.depth_sub1 = rospy.Subscriber(
            '/camera1/depth/image_rect_raw', Image, self.depth_callback, callback_args=0, queue_size=1
        )
        self.depth_sub2 = rospy.Subscriber(
            '/camera2/depth/image_rect_raw', Image, self.depth_callback, callback_args=1, queue_size=1
        )

        # Subscriber for command_task topic
        self.command_task_sub = rospy.Subscriber('/command_task', Float64MultiArray, self.command_task_callback)

        # Publisher for stalk coordinates
        self.stalk_position_pub = rospy.Publisher('/command', Float64MultiArray, queue_size=10)

        # Publisher for the segmented image (debugging, for two cameras)
        self.segmented_image_pubs = [
            rospy.Publisher('/camera1/segmented_image', Image, queue_size=10),
            rospy.Publisher('/camera2/segmented_image', Image, queue_size=10)
        ]

        # --- tf2 用のセットアップ ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # カメラ→base_link 変換用フレーム名
        self.camera_frames = {
            'top': 'camera1_color_optical_frame',
            'side': 'camera2_color_optical_frame'
        }
        self.target_frame = 'base_link'

    def load_model(self):
        """Load the pre-trained segmentation model"""
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('crop_recognition2')
        model_path = os.path.join(package_path, 'models', 'best_deeplabv3plus_resnet50_voc_os16.pth')

        model = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=3, output_stride=16)

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(self.device)
            model.eval()
            rospy.loginfo("Model loaded successfully")
        else:
            rospy.logerr(f"Model checkpoint not found at {model_path}")
        return model

    def _finish_cycle(self, reason: str = ""):
        """1サイクルを確実に終了する（フラグ/状態をリセット）"""
        self.run_detection = False
        for k in self.detection_results:
            self.detection_results[k]['processed'] = False
            self.detection_results[k]['timestamp'] = None
            self.detection_results[k]['positions'] = []
        # ラッチも解除
        self.processing_flags = {'top': False, 'side': False}
        self.integration_done = False  # ★追加
        if reason:
            rospy.loginfo("Detection cycle finished (%s).", reason)

    def command_task_callback(self, command_task):
        """コマンドコールバック - 統合検出用に修正"""
        if not command_task.data:
            rospy.logwarn("command_task is empty. Ignoring.")
            return

        if command_task.data[0] == 1.0:
            self.target_class = 1  # Estimate stalk for class 1 (crop)
        elif command_task.data[0] == 2.0:
            self.target_class = 2  # Estimate stalk for class 2 (weed)
        else:
            rospy.logwarn("Invalid command received, defaulting to class 2")
            self.target_class = 2

        self.run_detection = True
        rospy.loginfo(f"Target class set to {self.target_class}")

        # 統合検出用のリセット（各カメラ1フレーム勝負）
        self.detection_results = {
            'top':  {'positions': [], 'timestamp': None, 'processed': False},
            'side': {'positions': [], 'timestamp': None, 'processed': False}
        }
        self.processing_flags = {'top': False, 'side': False}
        self.integration_done = False  # ★追加
        
        rospy.loginfo(f"Starting integrated detection for class {self.target_class}")

    def image_callback(self, ros_image, camera_idx):
        """画像処理コールバック - 各サイクルで各カメラ最初の1フレームのみ処理"""
        if not self.run_detection:
            return

        camera_type = self.camera_types[camera_idx]

        # すでにこのカメラは処理済みなら即スキップ
        if self.detection_results[camera_type]['processed']:
            return

        # 同時多発の再入を禁止（ラッチ）
        if self.processing_flags[camera_type]:
            return

        # === この時点で「このカメラの最初の1フレーム」を確保 ===
        self.processing_flags[camera_type] = True
        rospy.loginfo(f"Image callback triggered for {camera_type} camera (locked)")

        try:
            positions = []

            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
            pred = self.predict(cv_image)
            segmented_image = self.decode_target(pred, cv_image)

            cv_image = cv_image.astype(np.uint8)
            segmented_image = segmented_image.astype(np.uint8)

            # カメラタイプ別の位置検出（ここで1フレーム分だけ抽出）
            if camera_type == 'top':
                positions = self.detect_pixel_coordinates(segmented_image, pred)
                self.detection_results['top']['positions'] = positions
            elif camera_type == 'side':
                positions = self.detect_z_centroid(segmented_image, pred)
                self.detection_results['side']['positions'] = positions

            # タイムスタンプ更新 & 1フレーム処理済みマーク
            self.detection_results[camera_type]['timestamp'] = rospy.Time.now()
            self.detection_results[camera_type]['processed'] = True

            rospy.loginfo(f"{camera_type} positions (1-frame): {positions}")

            # 可視化
            if positions:
                for pos in positions:
                    cv2.circle(segmented_image, (pos[0], pos[1]), 5, (0, 255, 255), -1)
            segmented_image_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding="bgr8")
            self.segmented_image_pubs[camera_idx].publish(segmented_image_msg)

            # 両カメラが1フレームずつ揃ったら統合へ
            self.try_integrate_coordinates()

        except Exception as e:
            rospy.logerr(f"Error in image_callback for {camera_type} camera: {e}")

        finally:
            # このフレームの処理は終わり（再入可能に戻す）
            self.processing_flags[camera_type] = False

    def depth_callback(self, ros_depth_image, camera_idx):
        """Depth画像処理コールバック - 今は保持のみ（必要に応じて拡張）"""
        if not self.run_detection:
            return
        try:
            _ = self.bridge.imgmsg_to_cv2(ros_depth_image, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr(f"Error converting depth image: {e}")

    def predict(self, cv_image):
        """Perform segmentation on the input image"""
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(input_tensor).max(1)[1].cpu().numpy()[0]
        return pred

    def decode_target(self, mask, original_image):
        """Decode segmentation class labels into a color image."""
        palette = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0]])  # 背景, クラス1, クラス2の色
        color_mask = np.zeros_like(original_image)
        for class_id in range(1, palette.shape[0]):
            color_mask[mask == class_id] = palette[class_id]
        color_mask[mask == 0] = original_image[mask == 0]
        return color_mask

    def detect_pixel_coordinates(self, cv_image, pred):
        """上からのカメラ：雑草/作物のピクセル座標を検出"""
        positions = []
        original_mask = (pred == self.target_class).astype(np.uint8) * 255

        # モルフォロジー処理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        closed_mask = cv2.erode(original_mask, kernel, iterations=1)
        opened_mask = cv2.dilate(closed_mask, kernel, iterations=3)

        contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 上カメラのROI設定（必要に応じて調整）
                roi_x_min = 340  # ハードに合わせて
                roi_x_max = roi_x_min + (1280 - 680)
                roi_y_min = 150
                roi_y_max = roi_y_min + (720 - 300)
                cv2.rectangle(cv_image, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (0, 255, 255), 2)

                if roi_x_min <= cX <= roi_x_max and roi_y_min <= cY <= roi_y_max:
                    positions.append((cX, cY))
                    rospy.loginfo(f"Top camera pixel coordinates: ({cX}, {cY})")

        return positions

    def detect_z_centroid(self, cv_image, pred):
        """横からのカメラ：Z軸の重心を検出（今は擬似Z）"""
        positions = []
        original_mask = (pred == self.target_class).astype(np.uint8) * 255

        # モルフォロジー処理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        closed_mask = cv2.erode(original_mask, kernel, iterations=1)
        opened_mask = cv2.dilate(closed_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 横カメラのROI設定（必要に応じて調整）
                roi_x_min = 340
                roi_x_max = roi_x_min + (1280 - 680)
                roi_y_min = 150
                roi_y_max = roi_y_min + (720 - 300)
                cv2.rectangle(cv_image, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (0, 255, 255), 2)

                if roi_x_min <= cX <= roi_x_max and roi_y_min <= cY <= roi_y_max:
                    z_centroid = self.calculate_z_centroid(cY)
                    positions.append((cX, cY, z_centroid))
                    rospy.loginfo(f"Side camera Z centroid: {z_centroid} at image_y={cY}")

        return positions

    def calculate_z_centroid(self, image_y):
        """横カメラの画像Y座標からZ軸重心を計算（仮）"""
        camera_height = 0.2         # m
        image_center_y = 360.0      # px
        pixel_to_meter_ratio = 0.001  # m/px

        z_relative = (image_center_y - image_y) * pixel_to_meter_ratio
        z_absolute = camera_height + z_relative

        return max(0, z_absolute)

    # --- 追加：カメラ座標から base_link へ座標変換するユーティリティ ---
    def transform_to_base(self, camera_type, x_cam, y_cam, z_cam):
        """
        カメラ座標系 (camera*_color_optical_frame) の点を base_link 座標系に変換する。
        x_cam, y_cam, z_cam は「カメラ光学座標系」での値（m）。
        """
        if camera_type not in self.camera_frames:
            rospy.logwarn("Unknown camera_type '%s' for TF transform", camera_type)
            return None

        frame_id = self.camera_frames[camera_type]

        point_in = PointStamped()
        point_in.header.stamp = rospy.Time.now()
        point_in.header.frame_id = frame_id
        point_in.point.x = float(x_cam)
        point_in.point.y = float(y_cam)
        point_in.point.z = float(z_cam)

        try:
            # target_frame ← source_frame
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                frame_id,
                point_in.header.stamp,
                rospy.Duration(0.1)
            )
            point_out = tf2_geometry_msgs.do_transform_point(point_in, transform)
            return point_out
        except Exception as e:
            rospy.logwarn("TF transform failed (%s -> %s): %s", frame_id, self.target_frame, e)
            return None

    def try_integrate_coordinates(self):
        if self.integration_done:
            return  # ★追加：二重publish防止
        """両カメラの結果を統合して送信。検出が空でもサイクルは必ず終了する。"""
        now = rospy.Time.now()
        top = self.detection_results['top']
        side = self.detection_results['side']

        # どちらか未処理なら待つ（1フレームずつ揃うのを待機）
        if not (top['processed'] and side['processed']):
            return
        
        ts_top = top['timestamp']
        ts_side = side['timestamp']

        # タイムスタンプが欠けていたら終了
        if ts_top is None or ts_side is None:
            self._finish_cycle(reason="timestamp missing")
            return

        # タイムアウト（保険）
        if ((now - top['timestamp']).to_sec() > self.integration_timeout or
                (now - side['timestamp']).to_sec() > self.integration_timeout):
            rospy.logwarn("Detection data timeout - finishing cycle without publish")
            self._finish_cycle(reason="timeout")
            return
            
        self.integration_done = True  # ★追加：最初の1回で確定
        top_positions = top['positions']        # [(x,y), ...]
        side_positions = side['positions']      # [(x,y,z), ...]

        sent_any = False

        # 上カメラ（ピクセル座標）: [x, y, z=-1, cam_id=0]
        if top_positions:
            for px, py in top_positions:
                msg = Float64MultiArray()
                msg.data = [float(px), float(py), -1.0, 0.0]
                self.stalk_position_pub.publish(msg)
            rospy.loginfo("Published %d TOP detections", len(top_positions))
            sent_any = True

        # 横カメラ（Z重心）: [x=-1, y=-1, z, cam_id=1]
        if side_positions:
            for _, _, zc in side_positions:
                # zc は「カメラ座標系のZ」とみなして、base_link に変換してみる
                base_point = self.transform_to_base('side', 0.0, 0.0, zc)
                if base_point is not None:
                    z_world = base_point.point.z
                    rospy.loginfo("Transformed Z (camera->base_link): %.3f -> %.3f",
                                  zc, z_world)
                else:
                    # 変換失敗時は元の値をそのまま使う
                    z_world = zc

                msg = Float64MultiArray()
                msg.data = [-1.0, -1.0, float(z_world), 1.0]
                self.stalk_position_pub.publish(msg)
            rospy.loginfo("Published %d SIDE detections", len(side_positions))
            sent_any = True
            
            
        px, py = top_positions[0]
        _, _, zc = side_positions[0]    
        
        msg = Float64MultiArray()
        # /command : [x, y, z, cam_id] として cam_id=2 を「統合結果」とみなす
        msg.data = [float(px), float(py), float(zc), 2.0]
        self.stalk_position_pub.publish(msg)

        rospy.loginfo(
            "Published integrated 3D stalk position (x=%.1f, y=%.1f, z=%.3f, cam_id=2)",
            px, py, zc
        )
        
        if not sent_any:
            rospy.logwarn("No detections from either camera.")
            # 検出が無くてもサイクルは終了
            self._finish_cycle(reason="done")
            rospy.loginfo("All stalks weeded!")
        else:
            rospy.loginfo("Weeding now! (finish after delay)")
            # 非ブロッキングで一定時間後にサイクル終了
            rospy.Timer(rospy.Duration(60.0), self._timer_finish_cycle_cb, oneshot=True)

    def _timer_finish_cycle_cb(self, _event):
        self._finish_cycle(reason="done")
        rospy.loginfo("All stalks weeded!")


if __name__ == '__main__':
    try:
        node = CropRecognitionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

