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
from torch.utils.data import DataLoader
from PIL import Image as PILImage
import sys
import time
import pyrealsense2 as rs

sys.path.append('/home/jetpack513/weed_ws2/src/crop_recognition2/src')
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

        # Initialize the run_detection flag here
        self.run_detection = False

        # **カメラタイプの定義**
        self.camera_types = {
            0: 'top',      # camera1 = 上からのカメラ（XY座標用）
            1: 'side'      # camera2 = 横からのカメラ（Z座標用）
        }
        
        # **各カメラからの検出結果を保存**
        self.detection_results = {
            'top': {'positions': [], 'timestamp': None},
            'side': {'positions': [], 'timestamp': None}
        }
        
        # **統合結果の保存**
        self.integrated_targets = []
        
        # **座標統合用のタイムアウト（秒）**
        self.integration_timeout = 2.0

        # --- Setup for two cameras ---
        # Camera 1 topics (Top camera)
        self.image_sub1 = rospy.Subscriber('/camera1/color/image_raw', Image, self.image_callback, callback_args=0)
        # Camera 2 topics (Side camera)
        self.image_sub2 = rospy.Subscriber('/camera2/color/image_raw', Image, self.image_callback, callback_args=1)

        # Subscriber for command_task topic
        self.command_task_sub = rospy.Subscriber('/command_task', Float64MultiArray, self.command_task_callback)

        # Publisher for stalk coordinates
        self.stalk_position_pub = rospy.Publisher('/command', Float64MultiArray, queue_size=10)

        # Publisher for the segmented image (debugging, for two cameras)
        self.segmented_image_pubs = [
            rospy.Publisher('/camera1/segmented_image', Image, queue_size=10),
            rospy.Publisher('/camera2/segmented_image', Image, queue_size=10)
        ]

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

    def command_task_callback(self, command_task):
        """コマンドコールバック - 統合検出用に修正"""
        if command_task.data[0] == 1.0:
            self.target_class = 1  # Estimate stalk for class 1 (crop)
        elif command_task.data[0] == 2.0:
            self.target_class = 2  # Estimate stalk for class 2 (weed)
        else:
            rospy.logwarn("Invalid command received, defaulting to class 2")
            self.target_class = 2
        
        self.run_detection = True
        
        # **統合検出用のリセット**
        self.detection_results = {
            'top': {'positions': [], 'timestamp': None},
            'side': {'positions': [], 'timestamp': None}
        }
        self.integrated_targets = []
        
        rospy.loginfo(f"Starting integrated detection for class {self.target_class}")

    def image_callback(self, ros_image, camera_idx):
        """画像処理コールバック - カメラタイプ別に処理を分岐"""
        if self.run_detection:
            camera_type = self.camera_types[camera_idx]
            rospy.loginfo(f"Image callback triggered for {camera_type} camera")
            
            try:
                cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
                pred = self.predict(cv_image)
                segmented_image = self.decode_target(pred, cv_image)

                cv_image = cv_image.astype(np.uint8)
                segmented_image = segmented_image.astype(np.uint8)

                # **カメラタイプ別の位置検出**
                if camera_type == 'top':
                    positions = self.detect_pixel_coordinates(segmented_image, pred)
                else:  # side camera
                    positions = self.detect_z_centroid(segmented_image, pred)
                
                # **検出結果をタイムスタンプ付きで保存**
                self.detection_results[camera_type] = {
                    'positions': positions,
                    'timestamp': rospy.Time.now()
                }
                
                rospy.loginfo(f"{camera_type} camera positions: {positions}")

                # デバッグ用に茎位置を描画しpublish
                if positions:
                    for pos in positions:
                        cv2.circle(segmented_image, (pos[0], pos[1]), 5, (0, 255, 255), -1)

                # Publish the segmented image for this camera
                segmented_image_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding="bgr8")
                self.segmented_image_pubs[camera_idx].publish(segmented_image_msg)

                # **両カメラの結果が揃ったら統合処理を実行**
                self.try_integrate_coordinates()

            except Exception as e:
                rospy.logerr(f"Error in image_callback for {camera_type} camera: {e}")

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
        """上からのカメラ：雑草のピクセル座標を検出"""
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
                
                # **上カメラのROI設定**
                roi_x_min, roi_x_max = 200, 1080
                roi_y_min, roi_y_max = 100, 620
                
                cv2.rectangle(cv_image, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (0, 255, 255), 2)
                
                if roi_x_min <= cX <= roi_x_max and roi_y_min <= cY <= roi_y_max:
                    # **ピクセル座標のみを保存**
                    positions.append((cX, cY))
                    rospy.loginfo(f"Top camera pixel coordinates: ({cX}, {cY})")
        
        return positions

    def detect_z_centroid(self, cv_image, pred):
        """横からのカメラ：Z軸の重心を検出"""
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
                
                # **横カメラのROI設定**
                roi_x_min, roi_x_max = 300, 980
                roi_y_min, roi_y_max = 200, 500
                
                cv2.rectangle(cv_image, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (255, 0, 255), 2)
                
                if roi_x_min <= cX <= roi_x_max and roi_y_min <= cY <= roi_y_max:
                    # **Z座標重心として保存**
                    z_centroid = self.calculate_z_centroid(cY)
                    positions.append((cX, cY, z_centroid))
                    rospy.loginfo(f"Side camera Z centroid: {z_centroid} at image_y={cY}")
        
        return positions

    def calculate_z_centroid(self, image_y):
        """横カメラの画像Y座標からZ軸重心を計算"""
        # 簡易的なZ座標計算（実際のカメラキャリブレーションに合わせて調整）
        camera_height = 0.2  # 横カメラの地面からの高さ（m）
        image_center_y = 360.0  # 画像中心のY座標
        pixel_to_meter_ratio = 0.001  # ピクセルからメートルへの変換比率
        
        z_relative = (image_center_y - image_y) * pixel_to_meter_ratio
        z_absolute = camera_height + z_relative
        
        return max(0, z_absolute)

    def try_integrate_coordinates(self):
        """両カメラからの検出結果を統合してROSで送信"""
        current_time = rospy.Time.now()
        
        # 両カメラのデータが有効かチェック
        top_data = self.detection_results['top']
        side_data = self.detection_results['side']
        
        if (top_data['timestamp'] is None or side_data['timestamp'] is None):
            return
        
        # タイムアウトチェック
        if ((current_time - top_data['timestamp']).to_sec() > self.integration_timeout or
            (current_time - side_data['timestamp']).to_sec() > self.integration_timeout):
            rospy.logwarn("Detection data timeout - skipping integration")
            return
        
        # **データが揃った場合の送信処理**
        if top_data['positions'] and side_data['positions']:
            # 上カメラ：ピクセル座標送信
            for pixel_pos in top_data['positions']:
                pixel_x, pixel_y = pixel_pos
                command = Float64MultiArray()
                command.data = [pixel_x, pixel_y, 0]  # カメラID 0 = 上カメラ
                self.stalk_position_pub.publish(command)
                rospy.loginfo(f"Published top camera pixel coordinates: ({pixel_x}, {pixel_y})")
                
            # 横カメラ：Z座標重心送信
            for z_pos in side_data['positions']:
                _, _, z_centroid = z_pos
                command = Float64MultiArray()
                command.data = [z_centroid, 1]  # カメラID 1 = 横カメラ
                self.stalk_position_pub.publish(command)
                rospy.loginfo(f"Published side camera Z centroid: {z_centroid}")
            
            # 処理完了
            self.run_detection = False
            rospy.loginfo("Coordinate transmission completed!")
            time.sleep(60)  # 除草作業待機時間


if __name__ == '__main__':
    try:
        node = CropRecognitionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
