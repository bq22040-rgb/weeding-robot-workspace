#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, Bool
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

class WeedingRecognitionNode:
    def __init__(self):
        rospy.init_node('weeding_recognition', anonymous=True)

        self.target_class = 2
        self.run_detection = False

        # --- side-only 用 ---
        self.side_candidates = []  # [(x, y, z)_base, ...]
        self.target_queue = []     # 確定リスト [(x, y, z)_base, ...]

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.target_frame = 'base_link'
        self.side_frame = 'camera2_color_optical_frame'  # side の入力座標系（=deproject後のカメラ座標を載せるフレーム）

        # Subscriber
        self.command_task_sub = rospy.Subscriber('/command_task', Float64MultiArray, self.command_task_callback)
        self.side_sub = rospy.Subscriber('/side_recognition/result', Float64MultiArray, self.side_result_callback)

        # M5からの完了通知
        self.done_sub = rospy.Subscriber('/weeding_done', Bool, self.weeding_done_callback)

        # Publisher
        self.recognition_cmd_pub = rospy.Publisher('/recognition_command', Float64MultiArray, queue_size=1)
        self.stalk_position_pub = rospy.Publisher('/command', Float64MultiArray, queue_size=10)

        rospy.loginfo("WeedingRecognition (SIDE-ONLY) started")

    def command_task_callback(self, msg):
        if not msg.data:
            return

        self.target_class = int(msg.data[0])

        # 状態リセット
        self.run_detection = True
        self.side_candidates = []
        self.target_queue = []

        # side_recognition に「1フレーム処理」を指示
        cmd = Float64MultiArray()
        cmd.data = [float(self.target_class), 1.0]
        self.recognition_cmd_pub.publish(cmd)
        rospy.loginfo(f"Start SIDE-ONLY recognition for class {self.target_class}")

    def side_result_callback(self, msg):
        if not self.run_detection:
            return

        data = list(msg.data)

        # 2Dしか返ってこないケース（[px, py]）は今回は扱わない
        if len(data) < 3 or (len(data) % 3) != 0:
            rospy.logwarn(f"SIDE-ONLY: invalid data length {len(data)} (need 3N). ignore.")
            return

        # camera座標[mm] → base_link座標[m] に変換して候補を作る
        self.side_candidates = self.process_incoming_list(data)

        if not self.side_candidates:
            rospy.logwarn("SIDE-ONLY: no valid candidates after TF transform.")
            return

        # そのまま確定ターゲットにする（必要なら追加フィルタ/ソートはここ）
        final_targets = self.side_candidates

        # 例：x昇順で送る（今の挙動に合わせる）
        final_targets.sort(key=lambda p: p[0])

        self.target_queue = final_targets
        self.run_detection = False

        rospy.loginfo(f"SIDE-ONLY SUCCESS: {len(final_targets)} weeds ready to send.")
        self.send_next_target()

    def process_incoming_list(self, data):
        """フラットなリスト [X1,Y1,Z1, X2,Y2,Z2...] を base_link に変換して返す"""
        results = []
        for i in range(0, len(data), 3):
            # side_recognition は mm を publish している前提
            x_m = data[i]   / 1000.0
            y_m = data[i+1] / 1000.0
            z_m = data[i+2] / 1000.0

            base_xyz = self.transform_point(self.side_frame, x_m, y_m, z_m)
            if base_xyz:
                results.append(base_xyz)
            else:
                rospy.logwarn_throttle(1.0, "SIDE-ONLY: TF transform failed (check TF tree).")
        return results

    def send_next_target(self):
        """キューから次の雑草を取り出してM5へ送信"""
        if not self.target_queue:
            rospy.loginfo("All targets completed.")
            return

        x, y, z = self.target_queue.pop(0)

        # 座標変換: ROS(base_link)[m] -> M5[mm] に合わせて調整（元のまま踏襲）
        msg = Float64MultiArray()
        msg.data = [-x * 1000.0, y * 1000.0, -z * 1000.0, 0.0]  # [X, Y, Z, Mode]
        self.stalk_position_pub.publish(msg)
        rospy.loginfo(f"Sending target to M5: {msg.data[:3]}")

    def weeding_done_callback(self, msg):
        """M5から除草完了(True)を受け取ったら次を送る"""
        if msg.data:
            rospy.loginfo("Received done signal from M5.")
            rospy.sleep(1.0)
            self.send_next_target()

    def transform_point(self, frame_id, x, y, z):
        point_in = PointStamped()
        point_in.header.stamp = rospy.Time(0)
        point_in.header.frame_id = frame_id
        point_in.point.x = x
        point_in.point.y = y
        point_in.point.z = z
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, rospy.Time(0), rospy.Duration(0.2)
            )
            p_out = tf2_geometry_msgs.do_transform_point(point_in, transform)
            return (p_out.point.x, p_out.point.y, p_out.point.z)
        except Exception:
            return None

if __name__ == '__main__':
    node = WeedingRecognitionNode()
    rospy.spin()

