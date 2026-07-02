#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import std_msgs.msg
import csv

def publish_data(filename):
    # ROSノードの初期化
    rospy.init_node('csv_pub', anonymous=True)
    pub = rospy.Publisher('/command', std_msgs.msg.Float64MultiArray, queue_size=10)

    rate = rospy.Rate(0.0125)  # 0.1 Hz

    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
        
            for row in reader:
            # 各要素をトリムし、数値に変換
                try:
                    data = [float(value.strip()) for value in row]
                    rate.sleep()
                    msg = std_msgs.msg.Float64MultiArray(data=data)
                    pub.publish(msg)
                    rospy.loginfo("Published: {}".format(msg.data))
#                rospy.loginfo(f"Published: {msg.data}")
                except ValueError as e:
                    rospy.logerr("Error converting data: {}".format(e))
#                rospy.logerr(f"Error converting data: {e}")

                rate.sleep()

    except IOError as e:
        rospy.logerr("Failed to open file: {}".format(e))

if __name__ == '__main__':
    try:
        # コマンドライン引数でファイル名を取得
        import sys
        if len(sys.argv) != 2:
            rospy.logerr("Usage: rosrun your_package csv_reader <filename>")
            sys.exit(1)

        publish_data(sys.argv[1])
    except rospy.ROSInterruptException:
        pass
