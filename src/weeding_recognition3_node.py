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

        # 俯瞰カメラのみ
        self.top_candidates = []
        self.target_queue = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.target_frame = 'base_link'
        self.top_camera_frame = 'camera1_color_optical_frame'

        # Subscriber
        self.command_task_sub = rospy.Subscriber(
            '/command_task',
            Float64MultiArray,
            self.command_task_callback
        )

        self.top_sub = rospy.Subscriber(
            '/top_recognition/result',
            Float64MultiArray,
            self.top_result_callback
        )

        self.done_sub = rospy.Subscriber(
            '/weeding_done',
            Bool,
            self.weeding_done_callback
        )

        # Publisher
        self.recognition_cmd_pub = rospy.Publisher(
            '/recognition_command',
            Float64MultiArray,
            queue_size=1
        )

        self.stalk_position_pub = rospy.Publisher(
            '/command',
            Float64MultiArray,
            queue_size=10
        )

        rospy.loginfo("WeedingRecognition TOP-ONLY mode started")

    def command_task_callback(self, msg):
        if not msg.data:
            return

        self.target_class = int(msg.data[0])

        # 状態リセット
        self.run_detection = True
        self.top_candidates = []
        self.target_queue = []

        # top_recognition に認識開始命令を送る
        cmd = Float64MultiArray()
        cmd.data = [float(self.target_class), 1.0]
        self.recognition_cmd_pub.publish(cmd)

        rospy.loginfo(f"Start TOP-ONLY recognition for class {self.target_class}")

    def top_result_callback(self, msg):
        if not self.run_detection:
            return

        # top_recognition から来た [X,Y,Z, X,Y,Z, ...] を base_link に変換
        self.top_candidates = self.process_top_list(msg.data)

        if not self.top_candidates:
            rospy.logwarn("TOP-ONLY: No valid top candidates.")
            return

        # 俯瞰のみなので、そのままターゲット確定
        self.target_queue = self.top_candidates

        # 必要なら並び順を調整
        # x座標順に送る
        self.target_queue.sort(key=lambda p: p[0])

        self.run_detection = False

        rospy.loginfo(f"TOP-ONLY SUCCESS: {len(self.target_queue)} weeds ready to send.")
        self.send_next_target()

    def process_top_list(self, data):
        """
        top_recognition からのデータを処理する。
        想定: [X1,Y1,Z1, X2,Y2,Z2, ...] 単位は mm
        """

        results = []

        if len(data) < 3:
            rospy.logwarn(
                "TOP-ONLY: Received pixel coords only or invalid data. "
                "Need 3D coordinates [X,Y,Z]."
            )
            return results

        if len(data) % 3 != 0:
            rospy.logwarn(
                f"TOP-ONLY: Invalid data length {len(data)}. "
                "Expected multiple of 3."
            )
            return results

        for i in range(0, len(data), 3):
            x_m = data[i] / 1000.0
            y_m = data[i + 1] / 1000.0
            z_m = data[i + 2] / 1000.0

            base_xyz = self.transform_point(
                self.top_camera_frame,
                x_m,
                y_m,
                z_m
            )

            if base_xyz:
                results.append(base_xyz)
                rospy.loginfo(
                    f"TOP-ONLY candidate base_link: "
                    f"x={base_xyz[0]:.3f}, y={base_xyz[1]:.3f}, z={base_xyz[2]:.3f}"
                )
            else:
                rospy.logwarn("TOP-ONLY: TF transform failed for one candidate.")

        return results

    def send_next_target(self):
        """
        キューから次の雑草を取り出して M5 へ送信
        """
        if not self.target_queue:
            rospy.loginfo("TOP-ONLY: All targets completed.")
            return

        x, y, z = self.target_queue.pop(0)

        msg = Float64MultiArray()

        # ROS(base_link) -> M5座標系
        # 既存コードと同じ変換
        msg.data = [
            -x * 1000.0,
             y * 1000.0,
            -z * 1000.0,
             0.0
        ]

        self.stalk_position_pub.publish(msg)

        rospy.loginfo(f"TOP-ONLY Sending target to M5: {msg.data[:3]}")

    def weeding_done_callback(self, msg):
        """
        M5から除草完了 True を受け取ったら次の雑草を送る
        """
        if msg.data:
            rospy.loginfo("TOP-ONLY: Received done signal from M5.")
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
                self.target_frame,
                frame_id,
                rospy.Time(0),
                rospy.Duration(0.1)
            )

            p_out = tf2_geometry_msgs.do_transform_point(point_in, transform)

            return (
                p_out.point.x,
                p_out.point.y,
                p_out.point.z
            )

        except Exception as e:
            rospy.logwarn(f"TOP-ONLY: TF transform error: {e}")
            return None


if __name__ == '__main__':
    try:
        node = WeedingRecognitionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass