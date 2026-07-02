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
        self.integration_timeout = 2.0 

        # --- 複数個体管理用のバッファ ---
        self.top_candidates = []  # [(x, y, z)_base, ...]
        self.side_candidates = [] # [(x, y, z)_base, ...]
        self.target_queue = []    # 紐付け済み確定リスト [(x, y, z), ...]

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.target_frame = 'base_link'
        self.camera_frames = {'top': 'camera1_color_optical_frame', 'side': 'camera2_color_optical_frame'}

        # Subscriber
        self.command_task_sub = rospy.Subscriber('/command_task', Float64MultiArray, self.command_task_callback)
        self.top_sub = rospy.Subscriber('/top_recognition/result', Float64MultiArray, self.top_result_callback)
        self.side_sub = rospy.Subscriber('/side_recognition/result', Float64MultiArray, self.side_result_callback)
        
        # M5からの完了通知 (M5側で除草が終わったら True を投げてもらう想定)
        self.done_sub = rospy.Subscriber('/weeding_done', Bool, self.weeding_done_callback)

        # Publisher
        self.recognition_cmd_pub = rospy.Publisher('/recognition_command', Float64MultiArray, queue_size=1)
        self.stalk_position_pub = rospy.Publisher('/command', Float64MultiArray, queue_size=10)

        rospy.loginfo("WeedingRecognition (Multi-Target Mode) started")

    def command_task_callback(self, msg):
        if not msg.data: return
        self.target_class = int(msg.data[0])
        
        # 状態リセット
        self.run_detection = True
        self.top_candidates = []
        self.side_candidates = []
        self.target_queue = []

        # 認識開始命令
        cmd = Float64MultiArray()
        cmd.data = [float(self.target_class), 1.0]
        self.recognition_cmd_pub.publish(cmd)
        rospy.loginfo(f"Start recognition for class {self.target_class}")

    def top_result_callback(self, msg):
        if not self.run_detection: return
        # 受信データを3つずつの組に分解してbase_linkへ変換
        self.top_candidates = self.process_incoming_list(msg.data, 'top')
        self.try_integrate()

    def side_result_callback(self, msg):
        if not self.run_detection: return
        self.side_candidates = self.process_incoming_list(msg.data, 'side')
        self.try_integrate()

    def process_incoming_list(self, data, camera_type):
        """フラットなリスト [X1,Y1,Z1, X2,Y2,Z2...] を変換して返す"""
        results = []
        for i in range(0, len(data), 3):
            x_m, y_m, z_m = data[i]/1000.0, data[i+1]/1000.0, data[i+2]/1000.0
            base_xyz = self.transform_point(self.camera_frames[camera_type], x_m, y_m, z_m)
            if base_xyz:
                results.append(base_xyz)
        return results

    def try_integrate(self):
        """TopとSideのデータを距離ベースで紐付け、キューを作る"""
        if not self.top_candidates or not self.side_candidates:
            # 片方でも空なら、まだ統合できないので戻る
            return

        final_targets = []
        used_side_indices = set()

        rospy.loginfo(f"--- Integration Start: Top({len(self.top_candidates)}) vs Side({len(self.side_candidates)}) ---")

        # enumerate を追加して、インデックス i を取得できるように修正しました
        for i, t_pos in enumerate(self.top_candidates):
            best_dist = 0.15  # 紐付け許容範囲（20cm）
            best_side_idx = -1

            for s_idx, s_pos in enumerate(self.side_candidates):
                if s_idx in used_side_indices: continue
                
                # XY平面上での距離を計算
                dist = np.linalg.norm(np.array(t_pos[:2]) - np.array(s_pos[:2]))
                
                # デバッグログ：各ペアの距離を表示
                rospy.loginfo(f"  Check Top[{i}]-Side[{s_idx}] dist: {dist:.3f}m")

                if dist < best_dist:
                    best_dist = dist
                    best_side_idx = s_idx
            
            if best_side_idx != -1:
                # 紐付け成功
                target = (t_pos[0], t_pos[1], self.side_candidates[best_side_idx][2])
                final_targets.append(target)
                used_side_indices.add(best_side_idx)
                rospy.loginfo(f"  => Match Found! Top[{i}] <-> Side[{best_side_idx}] (Dist: {best_dist:.3f}m)")

        if final_targets:
            final_targets.sort(key=lambda p: p[0])
            self.target_queue = final_targets
            self.run_detection = False
            rospy.loginfo(f"SUCCESS: {len(final_targets)} weeds ready to send.")
            self.send_next_target()
        else:
            # ログの出過ぎを防ぐため、1サイクルに1回だけ警告
            rospy.logwarn_throttle(1.0, "FAILED: No matching weeds found within 0.2m threshold.")

    def send_next_target(self):
        """キューから次の雑草を取り出してM5へ送信"""
        if not self.target_queue:
            rospy.loginfo("All targets completed.")
            return

        x, y, z = self.target_queue.pop(0)
        
        # 座標変換: ROS(base_link) -> M5の座標系に合わせて調整
        msg = Float64MultiArray()
        msg.data = [-x*1000.0, y*1000.0, -z*1000.0, 0.0]# [X, Y, Z, Mode]　[-x*1000.0, y*1000.0, -z*1000.0, 0.0]
        self.stalk_position_pub.publish(msg)
        rospy.loginfo(f"Sending target to M5: {msg.data[:3]}")

    def weeding_done_callback(self, msg):
        """M5から除草完了(True)を受け取ったら次を送る"""
        if msg.data:
            rospy.loginfo("Received done signal from M5.")
            # 少し待機してから次を送信（安全のため）
            rospy.sleep(1.0)
            self.send_next_target()

    def transform_point(self, frame_id, x, y, z):
        # (既存の transform_point 関数と同じため省略可)
        point_in = PointStamped()
        point_in.header.stamp = rospy.Time(0)
        point_in.header.frame_id = frame_id
        point_in.point.x, point_in.point.y, point_in.point.z = x, y, z
        try:
            transform = self.tf_buffer.lookup_transform(self.target_frame, frame_id, rospy.Time(0), rospy.Duration(0.1))
            p_out = tf2_geometry_msgs.do_transform_point(point_in, transform)
            return (p_out.point.x, p_out.point.y, p_out.point.z)
        except: return None

if __name__ == '__main__':
    node = WeedingRecognitionNode()
    rospy.spin()
