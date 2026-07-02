import os
import rospkg
import rospy
import cv2
import torch
import torch.nn as nn
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray,Float64 #comand_task用
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from torchvision import transforms as T
from torch.utils.data import DataLoader
from PIL import Image as PILImage
import sys#networkとutils用

sys.path.append('/home/auto-takanishi01/miniagri_ws/src/crop_recognition/src')

import network
# import utils
from tqdm import tqdm

class CropRecognitionNode:
    def __init__(self):
        rospy.init_node('crop_recognition_node', anonymous=True)
        
        # Define transformations
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        rospy.loginfo("Crop recognition node initialized")

        # Default class for stalk estimation (class 1: coriander)# 最終的にはcommand_taskを代入する

        # 茎publish関連
        self.target_class = 1
        self.stalk_position = []
        self.stalk_count = 0 # 送った茎の数

        # Depth image (initial value is None)
        self.depth_image = None

        # Set up the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

        self.bridge = CvBridge()
        
        # **Initialize the run_detection flag here** #一回だけ実行するようフラグ
        self.run_detection = False
        self.command_received = False  # command_taskが受信されたかどうか

        # Set up the subscriber to the Realsense camera
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

        # Subscriber for command_task topic
        self.command_task_sub = rospy.Subscriber('/command_task', Float64MultiArray, self.command_task_callback)
        self.task_state_sub = rospy.Subscriber('/task_state_topic', Float64, self.task_state_callback)

        # Publisher for stalk coordinates
        self.stalk_position_pub = rospy.Publisher('/stalk_position', Point, queue_size=10)

        # Publisher for the segmented image (debugging)
        self.segmented_image_pub = rospy.Publisher('/segmented_image', Image, queue_size=10)
        self.debug_depth_pub = rospy.Publisher('/depth_image', Image, queue_size=10)

    def load_model(self): #latest
        """Load the pre-trained segmentation model"""
        # Get the path to the crop_detection package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('crop_recognition')
        model_path = os.path.join(package_path, 'models', 'best_deeplabv3plus_resnet50_voc_os16.pth')

        # Use the specified model architecture
        model = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=3, output_stride=16)

        # Load model state from the checkpoint
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

    # def load_model(self):
    #     """Load the pre-trained segmentation model"""
    #     # Get the path to the crop_detection package
    #     rospack = rospkg.RosPack()
    #     package_path = rospack.get_path('crop_recognition')  # 'crop_detection' パッケージ名
    #     model_path = os.path.join(package_path, 'models', 'best_deeplabv3plus_resnet50_voc_os16.pth')  # モデルのパス

    #     model = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=3, output_stride=16)
    #     checkpoint = torch.load(model_path, map_location=self.device,weights_only=False)
    #     model.load_state_dict(checkpoint["model_state"])
    #     model = nn.DataParallel(model)
    #     model.to(self.device)
    #     model.eval()
    #     rospy.loginfo("Model loaded successfully")
    #     return model

    def command_task_callback(self, command_task):
        """Callback to handle commands for changing the target class"""
        if command_task.data[0] == 1.0:
            self.target_class = 1  # Estimate stalk for class 1 (coriander)
        elif command_task.data[0] == 2.0:
            self.target_class = 2  # Estimate stalk for class 2 (weed)
        else:
            rospy.logwarn("Invalid command received, defaulting to class 1")
            self.target_class = 1
        self.run_detection = True # 認識を行うフラグを立てる
        self.command_received = True  # コマンドを受信済みにする
        rospy.loginfo(f"Target class set to {self.target_class}")

    def image_callback(self, ros_image):
        """Callback to process incoming images from Realsense camera"""
        if self.run_detection:  # Run detection only if command_task was received
            rospy.loginfo("Image callback triggered") #debug
            try:
                # Convert ROS image to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')

                # Predict crop and weed segmentation
                pred = self.predict(cv_image)

                # Decode the prediction to a color image
                segmented_image = self.decode_target(pred,cv_image)

                # Ensure both images have the same data type (uint8)
                cv_image = cv_image.astype(np.uint8)
                segmented_image = segmented_image.astype(np.uint8)

                # # Visualize segmentation
                # segmented_image = self.visualize_segmentation(pred,cv_image)
                # segmented_image = self.visualize_segmentation(cv_image, pred)

                # Extract and save stalk position based on target class (but don't publish here)
                # self.detect_stalk_position(cv_image, pred)#now
                self.stalk_positions =self.detect_stalk_position(segmented_image, pred)
                
                # デバッグ用に茎位置を描画しpublish
                if self.stalk_positions:
                    for (cX, cY) in self.stalk_positions:
                        cv2.circle(segmented_image, (cX, cY), 5, (0, 255, 255), -1)  # Yellow dot for stalk

                # Publish the segmented image with the stalk position (ROS image message)
                segmented_image_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding="bgr8")
                self.segmented_image_pub.publish(segmented_image_msg)

                # 最初の茎だけpublish
                if self.stalk_positions:
                    self.publish_stalk_position()
                
                # Reset the flag so it runs only once per command
                rospy.loginfo("Detection completed, resetting run_detection to False")
                self.run_detection = False

            except Exception as e:
                rospy.logerr(f"Error in image_callback: {e}")
        # else:
        #     rospy.loginfo("run_detection is False, skipping image processing.")

    def predict(self, cv_image):
        """Perform segmentation on the input image"""
        # pil_image = PILImage.fromarray(cv_image)
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = self.model(input_tensor).max(1)[1].cpu().numpy()[0]  # HW
        # rospy.loginfo("Segmentation completed")
        return pred
        
    def decode_target(self, mask, original_image):#背景は塗りつぶさず、クラス1とクラス2だけ塗りつぶす
        """Decode segmentation class labels into a color image."""
        # Define the color palette for the classes (class 0: background, class 1: blue, class 2: green)
        palette = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0]])  # 背景, クラス1, クラス2の色

        # Create a blank image for the segmentation result
        color_mask = np.zeros_like(original_image)

        # Assign colors to each class, but leave the background class (class 0) unchanged
        for class_id in range(1, palette.shape[0]):  # Start from class 1 (skip class 0)
            color_mask[mask == class_id] = palette[class_id]

        # For background (class 0), retain the original image's pixels
        color_mask[mask == 0] = original_image[mask == 0]

        return color_mask

    # def decode_target(self,mask):#背景、クラス1、クラス2すべて塗りつぶす
    #     """Decode segmentation class labels into a color image."""
    #     palette = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0]])  # 背景, クラス1, クラス2の色
    #     return palette[mask]

    def visualize_segmentation(self, pred,cv_image):
        """Overlay the segmentation mask on the original image."""
        # predはモデルの出力クラスマップです
        decoded_segmentation = self.decode_target(pred)  # decode_targetでカラー化
        segmented_image = cv2.addWeighted(cv_image, 0.6, decoded_segmentation, 0.4, 0)
        return segmented_image 
    
    # def visualize_segmentation(self, cv_image, pred):
        # """Overlay the segmentation mask on the original image"""
        # # Create a color mask for visualization
        # color_mask = np.zeros_like(cv_image)
        
        # # Assign colors to different classes (background, coriander, weed)
        # color_mask[pred == 1] = [0,0 ,255]  # Red for coriander
        # color_mask[pred == 2] = [177, 0, 177]  # Blue for weed
        
        # # Overlay the mask onto the original image
        # segmented_image = cv2.addWeighted(cv_image, 0.6, color_mask, 0.4, 0)
        # return segmented_image

    def detect_stalk_position(self, cv_image, pred):
        """Detect stalk using morphology and publish its position"""
        positions = []
        # Extract red mask (class 1 is coriander)
        original_mask = (pred == self.target_class).astype(np.uint8) * 255

        # Apply morphology transformations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(original_mask, cv2.MORPH_CLOSE, kernel)
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

        # Find contours and compute centroid
        contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Define the inner region of interest (ROI) #範囲内か判定する場合
                # ROI dimensions: 880x470 in the center of the 1280x870 image
                roi_x_min = 340 #ハードに合わせて
                roi_x_max = roi_x_min + (1280-680)
                roi_y_min = 200
                roi_y_max = roi_y_min + (720-400)
                cv2.rectangle(cv_image, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (0, 255, 255), 2)  # Yellow rectangle for region
                # Check if the stalk position is within the defined ROI
                if roi_x_min <= cX <= roi_x_max and roi_y_min <= cY <= roi_y_max:
                    # Publish the centroid (stalk position) if it is within the ROI
                    positions.append((cX, cY))
                    rospy.loginfo(f"Stalk position within ROI: ({cX}, {cY})")

        return positions
    
    def depth_callback(self, depth_image_msg):
        """Callback to process depth image from Realsense camera"""
        try:
            # Convert ROS depth image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr(f"Error converting depth image: {e}")

    def publish_stalk_position(self):
        """Publish the stalk position with depth information."""
        if self.stalk_positions and self.stalk_count < len(self.stalk_positions):
            # 現在の茎の位置を取得
            cX, cY = self.stalk_positions[self.stalk_count]

            # スケールダウン
            x = int(cX / 1.5)
            y = int(cY / 1.5)

            # 150x150の領域を茎の位置の周りに定義
            x_min = max(0, x - 75)
            x_max = min(self.depth_image.shape[1], x + 75)
            y_min = max(0, y - 75)
            y_max = min(self.depth_image.shape[0], y + 75)

            # 深度画像からの領域を抽出して、最大深度を計算
            depth_window = self.depth_image[y_min:y_max, x_min:x_max]
            max_depth = np.max(depth_window)
            rospy.loginfo(f"Max depth in 100x100 region around stalk: {max_depth} meters")

            # 茎の位置を深度情報付きでpublish
            stalk_position = Point()
            stalk_position.x = (cX-160)/1.5 #640×480用に補正
            stalk_position.y = cY/1.5 #640×480用に補正
            stalk_position.z = max_depth
            self.stalk_position_pub.publish(stalk_position)

            #####デバッグ画像#####

            # Apply color map to depth image for visualization #カラーdebug画像用
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)

            cv2.circle(depth_colormap, (x, y), 10, (0, 0, 255), 3)  # Red circle at the (x, y) position
            # Optionally, draw a rectangle around the 400x400 region
            cv2.rectangle(depth_colormap, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red rectangle for region
            # Publish the debug image showing the point and region
            debug_depth_msg = self.bridge.cv2_to_imgmsg(depth_colormap, encoding="bgr8")

            self.debug_depth_pub.publish(debug_depth_msg)

            rospy.loginfo(f"Published stalk position (1280,720): ({cX}, {cY}, {max_depth})")

            # カウントを増やす
            self.stalk_count += 1

    def task_state_callback(self, task_state_msg):
        if task_state_msg.data==1:
            # task_state 受信ごとに次の茎の位置をpublish
            if self.stalk_count < len(self.stalk_positions):
                self.publish_stalk_position()
                rospy.loginfo(f"stalk_count:({self.stalk_count})")
            else:
                rospy.loginfo("No more stalk positions to publish.")
        else:
            rospy.loginfo("task_state_msg ERROR.")


if __name__ == '__main__':
    try:
        node = CropRecognitionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
