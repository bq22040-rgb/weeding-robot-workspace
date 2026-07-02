#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray, Bool
from visualization_msgs.msg import Marker, MarkerArray
from collections import deque

class WeedMarkerNode:
    def __init__(self):
        rospy.init_node("weed_marker_node")

        # ====== params ======
        self.frame_id = rospy.get_param("~frame_id", "base_link")
        self.scale_current = rospy.get_param("~scale_current", 0.035)
        self.scale_history = rospy.get_param("~scale_history", 0.02)
        self.keep_history = rospy.get_param("~keep_history", True)
        self.history_len = rospy.get_param("~history_len", 20)

        # ====== pubs/subs ======
        self.pub = rospy.Publisher("/weed_markers", MarkerArray, queue_size=1)

        rospy.Subscriber("/command", Float64MultiArray, self.cb_command)
        rospy.Subscriber("/weeding_done", Bool, self.cb_done)

        # ====== state ======
        self.history = deque(maxlen=self.history_len)
        self.has_current = False
        self.current_xyz = (0.0, 0.0, 0.0)

        rospy.loginfo("[weed_marker_node] ready. frame_id=%s", self.frame_id)

    # ---------- callbacks ----------
    def cb_command(self, msg):
        if len(msg.data) < 3:
            return

        x, y, z = msg.data[0], msg.data[1], msg.data[2]
        self.current_xyz = (x, y, z)
        self.has_current = True

        if self.keep_history:
            self.history.append((x, y, z))

        self.publish_markers()

    def cb_done(self, msg):
        # 除草完了 → 現在ターゲット削除
        if msg.data:
            self.has_current = False
            self.publish_markers(delete_current=True)

    # ---------- marker helpers ----------
    def make_sphere(self, mid, xyz, scale, rgba, ns):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
        m.pose.orientation.w = 1.0
        m.scale.x = scale
        m.scale.y = scale
        m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def make_delete(self, mid, ns):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = mid
        m.action = Marker.DELETE
        return m

    # ---------- publish ----------
    def publish_markers(self, delete_current=False):
        arr = MarkerArray()

        # 履歴（灰色・小）
        if self.keep_history:
            for i, xyz in enumerate(self.history):
                arr.markers.append(
                    self.make_sphere(
                        mid=100 + i,
                        xyz=xyz,
                        scale=self.scale_history,
                        rgba=(0.6, 0.6, 0.6, 0.6),
                        ns="history"
                    )
                )

        # 現在ターゲット（赤・大）
        if delete_current or not self.has_current:
            arr.markers.append(self.make_delete(mid=0, ns="current"))
        else:
            arr.markers.append(
                self.make_sphere(
                    mid=0,
                    xyz=self.current_xyz,
                    scale=self.scale_current,
                    rgba=(1.0, 0.0, 0.0, 1.0),
                    ns="current"
                )
            )

        self.pub.publish(arr)


if __name__ == "__main__":
    node = WeedMarkerNode()
    rospy.spin()

