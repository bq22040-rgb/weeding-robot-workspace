#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading
import rospy
from std_msgs.msg import Float64MultiArray, Float64, Bool
from dynamixel_sdk import *

# ==========================================================
# 基本設定
# ==========================================================

DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

ARM_IDS = [1, 4, 5]
GRIPPER_ID = 6
DXL_IDS = ARM_IDS + [GRIPPER_ID]

# ==========================================================
# モータ個別設定  動作MODE,角速度,角加速度,電流制限 等を設定
# ==========================================================

MOTOR_CONFIG = {

    1: {"OPERATING_MODE": 5, "GOAL_CURRENT": 800, "CURRENT_LIMIT": 1000,
        "PROFILE_ACCELERATION": 20, "PROFILE_VELOCITY": 100},

    4: {"OPERATING_MODE": 5, "GOAL_CURRENT": 800, "CURRENT_LIMIT": 1000,
        "PROFILE_ACCELERATION": 20, "PROFILE_VELOCITY": 100},

    5: {"OPERATING_MODE": 5, "GOAL_CURRENT": 800, "CURRENT_LIMIT": 1000,
        "PROFILE_ACCELERATION": 20, "PROFILE_VELOCITY": 100},

    # --- グリッパ用（XM430-W350） ---
    6: {"OPERATING_MODE": 5, "GOAL_CURRENT": 250, "CURRENT_LIMIT": 300,
        "PROFILE_ACCELERATION": 40, "PROFILE_VELOCITY": 200}
}

# ==========================================================
# Control Table
# ==========================================================

ADDR_OPERATING_MODE      = 11
ADDR_CURRENT_LIMIT       = 38
ADDR_TORQUE_ENABLE       = 64
ADDR_GOAL_CURRENT        = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY    = 112
ADDR_GOAL_POSITION       = 116
ADDR_PRESENT_POSITION    = 132

TORQUE_ENABLE  = 1
TORQUE_DISABLE = 0

# ==========================================================
# 角度変換
# ==========================================================

TICKS_PER_REV = 4096.0
DEG_TO_TICK   = TICKS_PER_REV / 360.0
TICK_TO_DEG   = 360.0 / TICKS_PER_REV

# ==========================================================
# 機構定数
# ==========================================================

HEIGHT_ADJUST       = 30        # 雑草を掴む時に縮めるZ軸の長さ
X_RADIUS            = 16.5      # x_radius
Y_RADIUS            = 25.0      # y_radius
YAW_GEAR_RATIO      = 2.0       # yaw_gear_ratio
PITCH_GEAR_RATIO    = 2.5       # pitch_gear_ratio
INITIAL_PITCH_ANGLE = 200    # INITIAL_PITCH_ANGLE　168.75(西早稲田居室調整済）
INITIAL_YAW_ANGLE   = 180      # INITIAL_YAW_ANGLE  242→280(西早稲田居室調整済）
ARM_LENGTH          = 183       # arm_length

PITCH_THIRTY_DEG    = 30.0 * PITCH_GEAR_RATIO + INITIAL_PITCH_ANGLE

# グリッパ開閉角度
# dmxdrv.py では相対角度で制御するため initial_pos_6 からのオフセットに変換する）
GRIPPER_CLOSE_ABS = 15.0   # dxl.setGoalPosition(6, 15, UNIT_DEGREE)
GRIPPER_OPEN_ABS  = 95.0   # dxl.setGoalPosition(6, 95, UNIT_DEGREE)

# ==========================================================
# ユーティリティ関数
# ==========================================================

def distance_angle_conv(distance, radius):
    """直線距離[mm] → 回転角度[deg] 変換 """
    return (360.0 * distance) / (2.0 * math.pi * radius)


# ==========================================================
# Controller
# ==========================================================

class DynamixelController:

    def __init__(self):

        rospy.init_node("dmxtest_multi_controller")

        # ---------- Dynamixel 接続 ----------
        self.portHandler   = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        self.portHandler.openPort()
        self.portHandler.setBaudRate(BAUDRATE)
        rospy.loginfo("Dynamixel接続OK")

        # ---------- 初期設定 ----------
        for dxl_id in DXL_IDS:
            cfg = MOTOR_CONFIG[dxl_id]
            self.write1(dxl_id, ADDR_TORQUE_ENABLE,       TORQUE_DISABLE)
            self.write1(dxl_id, ADDR_OPERATING_MODE,      cfg["OPERATING_MODE"])
            self.write2(dxl_id, ADDR_CURRENT_LIMIT,       cfg["CURRENT_LIMIT"])
            self.write4(dxl_id, ADDR_PROFILE_ACCELERATION, cfg["PROFILE_ACCELERATION"])
            self.write4(dxl_id, ADDR_PROFILE_VELOCITY,    cfg["PROFILE_VELOCITY"])
            self.write1(dxl_id, ADDR_TORQUE_ENABLE,       TORQUE_ENABLE)
            self.write2(dxl_id, ADDR_GOAL_CURRENT,        cfg["GOAL_CURRENT"])
            rospy.loginfo(f"ID{dxl_id} 初期設定完了")

        rospy.sleep(0.5)  # 安定待ち

        # ★ ID=1 初期位置を degree 換算で保存
        self.init_x_pos = self.read4(1, ADDR_PRESENT_POSITION) * TICK_TO_DEG
        rospy.loginfo(f"ID1 初期位置保存(init_x_pos): {self.init_x_pos:.2f} deg")

        # ★ ID=6 初期位置を ticks で保存（既存コードの相対角度制御用）
        self.initial_pos_6 = self.read4(GRIPPER_ID, ADDR_PRESENT_POSITION)
        rospy.loginfo(
            f"ID6 初期位置保存: {self.initial_pos_6 * TICK_TO_DEG:.2f} deg"
        )

        # ---------- 状態変数 ----------
        self.start_pos    = 30
        self.z_correction = 35

        self.z_t          = 0.0
        self.x_t_scaled   = 0.0
        self.y_t_scaled   = 0.0
        self.degree_yaw   = 0.0
        self.move_y       = 0.0
        self.mode         = -1
        self.flag_xyz     = 0
        self.yaw_step_count = 0

        # ---------- スレッドロック ----------
        # コールバック（ROSスピンスレッド）とメインループが
        # 共有変数を同時に読み書きしないよう排他制御する
        self.lock = threading.Lock()

        # ---------- ROS Publisher ----------
        # /task_state : 各コマンド完了後に状態値をpublishする
        self.task_state_pub = rospy.Publisher(
            "/task_state", Float64, queue_size=1
        )
        self.task_state_val = 0.0   # publish する値

        self.weeding_done_pub = rospy.Publisher(
            "/weeding_done", Bool, queue_size=1
        )

        # /z_command : Z軸制御ノード向けコマンドをpublishする

        self.z_command_pub = rospy.Publisher(
            "/z_command", Float64MultiArray, queue_size=1
#           "/z_command", Float64MultiArray, queue_size=1
        )
        # ---------- ROS Subscriber ----------
        rospy.Subscriber("/command", Float64MultiArray, self.callback)

        rospy.loginfo("DynamixelController 初期化完了 — メインループ開始")

        # ---------- メインループ ----------
        self._main_loop()

    # ======================================================
    # メインループ
    # flag_xyz==1 のとき、modeに対応した動作を行う
    # nh.spinOnce() の代替として rospy.sleep(0.01) を使用し、
    # ROSのコールバック処理を継続させる
    # ======================================================
    def _main_loop(self):

        rate = rospy.Rate(50)   # 50Hz ポーリング

        while not rospy.is_shutdown():

            # ロックを取得して flag_xyz とパラメータをスナップショット
            with self.lock:
                if self.flag_xyz == 1:
                    mode        = self.mode
                    self.flag_xyz = 0   # 受け取ったのでリセット
                else:
                    mode = -1   # 実行なし

            if mode == 0:
                self.cmd_remove_weed()
            elif mode == 1:
                self.cmd_return()
            elif mode == 2:
                self.cmd_adjust()
            elif mode == 11:
                self.cmd_yaw_plus_turn()
            elif mode == 12:
                self.cmd_yaw_minus_turn()
            elif mode == 13:
                self.cmd_pitchup_zdown()

            if mode in (0, 1, 2, 11, 12, 13):
                self._publish_task_state()
                rospy.sleep(0.3)

            rate.sleep()

    # ======================================================
    # Callback  (/command の受信処理)
    # ======================================================
    def callback(self, msg):
        mode = 30
        # -----------------------------------------------------------------------
        # (1) data[4] == 20 の時: data[0-3]を y,yaw,pitch,grip と認識しアーム操作
        # -----------------------------------------------------------------------
        if len(msg.data) >= 5 and msg.data[4] == 20:

            for i, dxl_id in enumerate(ARM_IDS):
                p_deg     = msg.data[i]
                goal_tick = int(p_deg * DEG_TO_TICK)
                self.write4(dxl_id, ADDR_GOAL_POSITION, goal_tick)

            # ID=6 は相対角度制御
            relative_deg  = msg.data[3]
            goal_tick_6   = int(self.initial_pos_6 + relative_deg * DEG_TO_TICK)
            self.write4(GRIPPER_ID, ADDR_GOAL_POSITION, goal_tick_6)

            present_tick = self.read4(GRIPPER_ID, ADDR_PRESENT_POSITION)
            present_deg  = present_tick * TICK_TO_DEG
            rospy.loginfo(
                f"[mode20] ID6 Relative:{relative_deg:.1f}deg  "
                f"Present:{present_deg:.1f}deg"
            )
            return

        # --------------------------------------------------------------------
        # (2) data[4] != 20 または len == 4 の時
        #     msg.data[y,x,z,mode] のy,xから　move_y,yawを算出してアームを操作
        # --------------------------------------------------------------------
        if len(msg.data) < 4:
            rospy.logwarn("data長が不足しています。無視します。")
            return

        if len(msg.data) == 4:
            mode = int(msg.data[3])
            start_pos    = self.start_pos
            z_correction = self.z_correction
            z_t          = start_pos + msg.data[2] - 140 - z_correction
            x_t_scaled   = float(msg.data[1])
            y_t_scaled   = float(msg.data[0])

            rospy.loginfo(
                f"[callback] mode={mode}  "
                f"x_t_scaled={x_t_scaled:.2f}  "
                f"y_t_scaled={y_t_scaled:.2f}  "
                f"z_t={z_t:.2f}"
            )

        # -------------------------------------------------------------
        # (3) mode==0,1,2,11,12,13 のとき x_t_scaled, y_t_scaled を補正
        # 
        # -------------------------------------------------------------
        if mode in (0, 1, 2, 11, 12, 13):
            d   = 40.0
            yaw = math.acos(
                min(abs(x_t_scaled) / (ARM_LENGTH + d), 1.0)
            )
            debug_deg = math.degrees(yaw)
            rospy.loginfo(f"[補正] 補正前yaw={debug_deg:.2f}deg")

            if x_t_scaled >= 0 and y_t_scaled >= 0:
                x_t_scaled -= d * math.cos(yaw)
                y_t_scaled -= d * math.sin(yaw)
            elif x_t_scaled >= 0 and y_t_scaled < 0:
                x_t_scaled -= d * math.cos(yaw)
                y_t_scaled += d * math.sin(yaw)
            elif x_t_scaled < 0 and y_t_scaled >= 0:
                x_t_scaled += d * math.cos(yaw)
                y_t_scaled -= d * math.sin(yaw)
            elif x_t_scaled < 0 and y_t_scaled < 0:
                x_t_scaled += d * math.cos(yaw)
                y_t_scaled += d * math.sin(yaw)

            rospy.loginfo(
                f"[補正] 補正後 x_t_scaled={x_t_scaled:.2f}  "
                f"y_t_scaled={y_t_scaled:.2f}"
            )

        # ----------------------------------------
        # (4) degree_yaw, move_y の算出
        # 
        # ----------------------------------------
            radian_yaw = 0.0
            degree_yaw = 0.0
            move_y     = 0.0

            if y_t_scaled >= 0 and x_t_scaled >= 0:
                radian_yaw = math.acos(
                    min(x_t_scaled / ARM_LENGTH, 1.0)
                )
                degree_yaw = math.degrees(radian_yaw)
                move_y     = y_t_scaled - ARM_LENGTH * math.sin(radian_yaw)

            elif y_t_scaled >= 0 and x_t_scaled < 0:
                radian_yaw = math.acos(
                    min(-x_t_scaled / ARM_LENGTH, 1.0)
                )
                degree_yaw = 180.0 - math.degrees(radian_yaw)
                move_y     = y_t_scaled - ARM_LENGTH * math.sin(radian_yaw)

            elif y_t_scaled < 0 and x_t_scaled >= 0:
                radian_yaw = math.acos(
                    min(x_t_scaled / ARM_LENGTH, 1.0)
                )
                degree_yaw = -math.degrees(radian_yaw)
                move_y     = y_t_scaled + ARM_LENGTH * math.sin(radian_yaw)

            elif y_t_scaled < 0 and x_t_scaled < 0:
                radian_yaw = math.acos(
                    min(-x_t_scaled / ARM_LENGTH, 1.0)
                )
                degree_yaw = -180.0 + math.degrees(radian_yaw)
                move_y     = y_t_scaled + ARM_LENGTH * math.sin(radian_yaw)

            rospy.loginfo(
                f"[算出] degree_yaw={degree_yaw:.2f}  move_y={move_y:.2f}"
            )

        # ----------------------------------------
        # (6) 可動範囲チェック
        # 
        # ----------------------------------------
            if abs(x_t_scaled) > ARM_LENGTH or move_y > 270 or move_y < -220:
                rospy.logwarn(
                    f"out of range  x_t_scaled={x_t_scaled:.2f}  "
                    f"move_y={move_y:.2f}"
                )
                return  # 処理終了

        # ----------------------------------------
        # 全パラメータをロックで保護しながら共有変数へ書き込み
        # flag_xyz = 1 をセット → メインループがコマンドを実行
        # ----------------------------------------
            with self.lock:
                self.mode        = mode
                self.z_t         = z_t
                self.x_t_scaled  = x_t_scaled
                self.y_t_scaled  = y_t_scaled
                self.degree_yaw  = degree_yaw
                self.move_y      = move_y
                self.flag_xyz    = 1

    # ======================================================
    # (5) Move_to_target
    #     Z軸制御ノード向けに /z_command をpublishする
    # 
    # ======================================================
    def move_to_target(self, z_target_mm):
        msg = Float64MultiArray()
        msg.data = [float(z_target_mm),0.0, 0.0, 0.0, 30.0]
#       msg.data = [0.0, 0.0, float(z_target_mm), 30.0]
        self.z_command_pub.publish(msg)
        rospy.loginfo(f"[Move_to_target] /z_command publish z={z_target_mm}")
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討
    # ======================================================
    # (8) task_state publish ヘルパー
    # ======================================================
    def _publish_task_state(self):
        msg = Float64()
        msg.data = self.task_state_val
        self.task_state_pub.publish(msg)
        rospy.loginfo(f"[task_state] publish {self.task_state_val}")
    
    def _publish_weeding_done(self):
        msg = Bool()
        msg.data = True
        self.weeding_done_pub.publish(msg)
        rospy.loginfo("[weeding_done] publish True")

    # ======================================================
    # グリッパ操作ヘルパー
    # 絶対角度[deg] を initial_pos_6 からの相対角度に変換して書き込む
    # ======================================================
    def _set_gripper_abs_deg(self, abs_deg):
        """main.cpp の dxl.setGoalPosition(6, abs_deg, UNIT_DEGREE) に相当"""
        relative_deg = abs_deg - (self.initial_pos_6 * TICK_TO_DEG)
        goal_tick    = int(self.initial_pos_6 + relative_deg * DEG_TO_TICK)
        self.write4(GRIPPER_ID, ADDR_GOAL_POSITION, goal_tick)
        rospy.loginfo(
            f"[Gripper] abs={abs_deg:.1f}deg  "
            f"relative={relative_deg:.1f}deg  "
            f"tick={goal_tick}"
        )

    # ======================================================
    # 各モータへ degree 指定で goal position を書き込む
    # 
    # ======================================================
    def _set_goal_deg(self, dxl_id, deg):
        goal_tick = int(deg * DEG_TO_TICK)
        self.write4(dxl_id, ADDR_GOAL_POSITION, goal_tick)

    # ======================================================
    # コマンド関数群 
    # 
    # ======================================================

    # --- mode == 0: cmdRemoveWeed ---
    def cmd_remove_weed(self):
        rospy.loginfo("[cmd_remove_weed] 開始")
        with self.lock:
            degree_yaw = self.degree_yaw
            move_y     = self.move_y
            z_t        = self.z_t

        yaw_goal    = degree_yaw * YAW_GEAR_RATIO + INITIAL_YAW_ANGLE
        y_axis_goal = -distance_angle_conv(move_y, X_RADIUS) + self.init_x_pos

        # Pitch を30deg 位置へ
        self._set_goal_deg(5, PITCH_THIRTY_DEG)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討

        # Z を手前まで伸ばす
        self.move_to_target(z_t - HEIGHT_ADJUST)

        # Yaw / Y軸 移動
        self._set_goal_deg(4, yaw_goal)
        self._set_goal_deg(1, y_axis_goal)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討

        # Z をターゲットへ
        self.move_to_target(z_t)

        # グリッパ閉
        rospy.sleep(0.2)
        self._set_gripper_abs_deg(GRIPPER_CLOSE_ABS)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討
        rospy.sleep(0.8)

        # 引き上げ (Z を start_pos へ)
        with self.lock:
            self.z_t = self.start_pos
        self.move_to_target(self.start_pos)

        # 初期 Yaw / Y 位置へ戻る
        self._set_goal_deg(4, INITIAL_YAW_ANGLE)
        self._set_goal_deg(1, self.init_x_pos)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討

        # 初期 Pitch へ戻る
        self._set_goal_deg(5, INITIAL_PITCH_ANGLE)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討

        # グリッパ開
        self._set_gripper_abs_deg(GRIPPER_OPEN_ABS)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討

        self.task_state_val = 0.0
        self._publish_weeding_done()
        rospy.loginfo("[cmd_remove_weed] 完了")

    # --- mode == 1: cmdReturn ---
    def cmd_return(self):
        rospy.loginfo("[cmd_return] 開始")
        with self.lock:
            self.z_t = self.start_pos

        self.move_to_target(self.start_pos)
        self._set_goal_deg(4, INITIAL_YAW_ANGLE)
        self._set_goal_deg(1, self.init_x_pos)
        self._set_goal_deg(5, INITIAL_PITCH_ANGLE)
        self._set_gripper_abs_deg(GRIPPER_OPEN_ABS)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討

        self.task_state_val = 0.0
        rospy.loginfo("[cmd_return] 完了")

    # --- mode == 2: cmdAdjust ---
    def cmd_adjust(self):
        rospy.loginfo("[cmd_adjust] 開始")
        with self.lock:
            degree_yaw = self.degree_yaw
            move_y     = self.move_y
            z_t        = self.z_t

        yaw_goal    = degree_yaw * YAW_GEAR_RATIO + INITIAL_YAW_ANGLE
        y_axis_goal = -distance_angle_conv(move_y, X_RADIUS) + self.init_x_pos

        self._set_goal_deg(5, PITCH_THIRTY_DEG)
        self._set_goal_deg(4, yaw_goal)
        self._set_goal_deg(1, y_axis_goal)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討

        rospy.sleep(0.2)
        self.move_to_target(z_t)

        self.task_state_val = 2.0
        rospy.loginfo("[cmd_adjust] 完了")

    # --- mode == 11: cmdYawPlusTurn ---
    def cmd_yaw_plus_turn(self):
        rospy.loginfo("[cmd_yaw_plus_turn] 開始")
        with self.lock:
            degree_yaw = self.degree_yaw
        self.yaw_step_count += 1
        target_yaw = (
            (degree_yaw + self.yaw_step_count * 3) * YAW_GEAR_RATIO
            + INITIAL_YAW_ANGLE
        )
        self._set_goal_deg(4, target_yaw)
        self.task_state_val = 2.0
        rospy.loginfo(
            f"[cmd_yaw_plus_turn] yaw_step={self.yaw_step_count}  "
            f"target_yaw={target_yaw:.2f}deg"
        )

    # --- mode == 12: cmdYawMinusTurn ---
    def cmd_yaw_minus_turn(self):
        rospy.loginfo("[cmd_yaw_minus_turn] 開始")
        with self.lock:
            degree_yaw = self.degree_yaw
        self.yaw_step_count -= 1
        target_yaw = (
            (degree_yaw + self.yaw_step_count * 3) * YAW_GEAR_RATIO
            + INITIAL_YAW_ANGLE
        )
        self._set_goal_deg(4, target_yaw)
        self.task_state_val = 2.0
        rospy.loginfo(
            f"[cmd_yaw_minus_turn] yaw_step={self.yaw_step_count}  "
            f"target_yaw={target_yaw:.2f}deg"
        )

    # --- mode == 13: cmdPitchupZdown ---
    def cmd_pitchup_zdown(self):
        rospy.loginfo("[cmd_pitchup_zdown] 開始")
        with self.lock:
            z_t = self.z_t

        # Pitch を38deg 位置へ
        self._set_goal_deg(5, 38.0 * PITCH_GEAR_RATIO + INITIAL_PITCH_ANGLE)
        # Z をターゲット+39mm へ
        self.move_to_target(int(z_t) + 54)

        # グリッパ閉
        rospy.sleep(0.2)
        self._set_gripper_abs_deg(GRIPPER_CLOSE_ABS)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討
        rospy.sleep(0.8)

        # Z を start_pos へ
        with self.lock:
            self.z_t = self.start_pos
        self.move_to_target(self.start_pos)

        # 初期 Yaw / Y へ
        self._set_goal_deg(4, INITIAL_YAW_ANGLE)
        self._set_goal_deg(1, self.init_x_pos)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討

        # 初期 Pitch へ
        self._set_goal_deg(5, INITIAL_PITCH_ANGLE)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討

        # グリッパ開
        self._set_gripper_abs_deg(GRIPPER_OPEN_ABS)
        rospy.sleep(2.0)    # waitUntilAllReachedへの変更も検討

        self.yaw_step_count = 0
        self.task_state_val = 0.0
        rospy.loginfo("[cmd_pitchup_zdown] 完了")

    # ======================================================
    # SDK ヘルパー
    # ======================================================
    def write1(self, dxl_id, addr, value):
        self.packetHandler.write1ByteTxRx(
            self.portHandler, dxl_id, addr, value)

    def write2(self, dxl_id, addr, value):
        self.packetHandler.write2ByteTxRx(
            self.portHandler, dxl_id, addr, value)

    def write4(self, dxl_id, addr, value):
        self.packetHandler.write4ByteTxRx(
            self.portHandler, dxl_id, addr, value)

    def read4(self, dxl_id, addr):
        value, _, _ = self.packetHandler.read4ByteTxRx(
            self.portHandler, dxl_id, addr)
        return value


if __name__ == "__main__":
    DynamixelController()
