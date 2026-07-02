#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, Bool
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class WeedingRecognitionNode3CamHungarian:
    def __init__(self):
        rospy.init_node('weeding_recognition_3cam_hungarian', anonymous=True)

        self.target_class = 2
        self.run_detection = False
        self.integration_done = False

        # 認識開始から統合まで少し待つ。
        # top/side/third の結果が出そろう前に確定するのを防ぐ。
        self.integration_timeout = rospy.get_param('~integration_timeout', 2.0)
        self.integration_started_at = None
        self.integration_timer = None

        # --- マッチングしきい値 [m] ---
        self.top_side_threshold = rospy.get_param('~top_side_threshold', 0.15)
        self.third_side_threshold = rospy.get_param('~third_side_threshold', 0.20)
        self.top_third_threshold = rospy.get_param('~top_third_threshold', 0.12)

        # third が存在するのに top だけしか合わない場合、かなり近い top だけ採用する。
        # これにより近接俯瞰の誤認識をある程度抑える。
        self.strict_top_threshold = rospy.get_param('~strict_top_threshold', 0.08)

        # top と third が矛盾した場合、全体俯瞰を優先するか。
        self.prefer_third_on_conflict = rospy.get_param('~prefer_third_on_conflict', True)

        # --- 複数個体管理用のバッファ ---
        self.top_candidates = []    # 近接俯瞰 camera1 [(x, y, z)_base, ...]
        self.side_candidates = []   # 横 camera2       [(x, y, z)_base, ...]
        self.third_candidates = []  # 全体俯瞰 camera3 [(x, y, z)_base, ...]
        self.target_queue = []      # 確定リスト [(x, y, z), ...]

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.target_frame = 'base_link'
        self.camera_frames = {
            'top': 'camera1_color_optical_frame',       # 近接俯瞰
            'side': 'camera2_color_optical_frame',      # 横
            'third': 'camera3_color_optical_frame'      # 全体俯瞰
        }

        # Subscriber
        self.command_task_sub = rospy.Subscriber(
            '/command_task',
            Float64MultiArray,
            self.command_task_callback,
            queue_size=1
        )

        self.top_sub = rospy.Subscriber(
            '/top_recognition/result',
            Float64MultiArray,
            self.top_result_callback,
            queue_size=1
        )

        self.side_sub = rospy.Subscriber(
            '/side_recognition/result',
            Float64MultiArray,
            self.side_result_callback,
            queue_size=1
        )

        self.third_sub = rospy.Subscriber(
            '/third_recognition/result',
            Float64MultiArray,
            self.third_result_callback,
            queue_size=1
        )

        # M5からの完了通知
        self.done_sub = rospy.Subscriber(
            '/weeding_done',
            Bool,
            self.weeding_done_callback,
            queue_size=1
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

        rospy.loginfo('WeedingRecognition 3Cam Hungarian started')
        rospy.loginfo(
            'thresholds: top-side=%.2fm, third-side=%.2fm, top-third=%.2fm, strict-top=%.2fm',
            self.top_side_threshold,
            self.third_side_threshold,
            self.top_third_threshold,
            self.strict_top_threshold
        )

        if not SCIPY_AVAILABLE:
            rospy.logwarn(
                'scipy is not installed. Use exact brute-force assignment for small candidate counts. '
                'For normal use, install scipy: sudo apt install python3-scipy'
            )

    # ============================================================
    # 認識開始
    # ============================================================
    def command_task_callback(self, msg):
        if not msg.data:
            return

        self.target_class = int(msg.data[0])

        # 状態リセット
        self.run_detection = True
        self.integration_done = False
        self.integration_started_at = rospy.Time.now()
        self.top_candidates = []
        self.side_candidates = []
        self.third_candidates = []
        self.target_queue = []

        # 古いタイマーが残っていれば停止
        if self.integration_timer is not None:
            try:
                self.integration_timer.shutdown()
            except Exception:
                pass

        # 認識開始から一定時間後に統合を試す
        self.integration_timer = rospy.Timer(
            rospy.Duration(self.integration_timeout),
            self.integration_timer_callback,
            oneshot=True
        )

        # 各認識ノードへ認識開始命令
        cmd = Float64MultiArray()
        cmd.data = [float(self.target_class), 1.0]
        self.recognition_cmd_pub.publish(cmd)

        rospy.loginfo(
            'Start recognition for class %d. Wait %.2fs before integration.',
            self.target_class,
            self.integration_timeout
        )

    def integration_timer_callback(self, _event):
        self.try_integrate(force=True)

    # ============================================================
    # 各カメラの結果受信
    # ============================================================
    def top_result_callback(self, msg):
        if not self.run_detection:
            return
        self.top_candidates = self.process_incoming_list(msg.data, 'top')
        rospy.loginfo('Received Top candidates: %d', len(self.top_candidates))
        self.try_integrate(force=False)

    def side_result_callback(self, msg):
        if not self.run_detection:
            return
        self.side_candidates = self.process_incoming_list(msg.data, 'side')
        rospy.loginfo('Received Side candidates: %d', len(self.side_candidates))
        self.try_integrate(force=False)

    def third_result_callback(self, msg):
        if not self.run_detection:
            return
        self.third_candidates = self.process_incoming_list(msg.data, 'third')
        rospy.loginfo('Received Third candidates: %d', len(self.third_candidates))
        self.try_integrate(force=False)

    def process_incoming_list(self, data, camera_type):
        """フラットなリスト [X1,Y1,Z1, X2,Y2,Z2...] [mm] を base_link [m] に変換する。"""
        results = []

        if len(data) < 3:
            rospy.logwarn(
                '%s result has less than 3 values. It may be pixel-only data. Skip.',
                camera_type
            )
            return results

        if len(data) % 3 != 0:
            rospy.logwarn(
                '%s result length is not multiple of 3: len=%d. Extra values will be ignored.',
                camera_type,
                len(data)
            )

        usable_len = (len(data) // 3) * 3

        for i in range(0, usable_len, 3):
            x_m = data[i] / 1000.0
            y_m = data[i + 1] / 1000.0
            z_m = data[i + 2] / 1000.0

            base_xyz = self.transform_point(
                self.camera_frames[camera_type],
                x_m,
                y_m,
                z_m
            )

            if base_xyz is not None:
                results.append(base_xyz)

        return results

    # ============================================================
    # ハンガリアン法によるマッチング
    # ============================================================
    def xy_distance(self, p1, p2):
        return float(np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2])))

    def solve_assignment(self, cost_matrix):
        """
        距離行列に対して、全体距離が最小になる割当を返す。
        scipy があれば Hungarian 法、なければ候補数が少ない前提で全探索する。
        """
        if SCIPY_AVAILABLE:
            return linear_sum_assignment(cost_matrix)

        # scipy がない場合の厳密な全探索。
        # 候補数が多いと重いので、通常運用では scipy 推奨。
        from itertools import permutations

        n_rows, n_cols = cost_matrix.shape

        if n_rows == 0 or n_cols == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        best_cost = None
        best_rows = None
        best_cols = None

        if n_rows <= n_cols:
            rows = list(range(n_rows))
            for cols in permutations(range(n_cols), n_rows):
                total = sum(cost_matrix[r, c] for r, c in zip(rows, cols))
                if best_cost is None or total < best_cost:
                    best_cost = total
                    best_rows = rows
                    best_cols = list(cols)
        else:
            cols = list(range(n_cols))
            for rows in permutations(range(n_rows), n_cols):
                total = sum(cost_matrix[r, c] for r, c in zip(rows, cols))
                if best_cost is None or total < best_cost:
                    best_cost = total
                    best_rows = list(rows)
                    best_cols = cols

        return np.array(best_rows, dtype=int), np.array(best_cols, dtype=int)

    def hungarian_match(self, primary_candidates, side_candidates, threshold, primary_label):
        """
        primary_candidates と side_candidates を XY距離でハンガリアンマッチングする。
        戻り値: [{primary_idx, side_idx, dist}, ...]
        """
        matches = []

        n_primary = len(primary_candidates)
        n_side = len(side_candidates)

        if n_primary == 0 or n_side == 0:
            return matches

        cost_matrix = np.zeros((n_primary, n_side), dtype=np.float32)

        rospy.loginfo(
            '--- Hungarian Matching: %s(%d) vs Side(%d), threshold=%.2fm ---',
            primary_label,
            n_primary,
            n_side,
            threshold
        )

        for i, p_pos in enumerate(primary_candidates):
            for j, s_pos in enumerate(side_candidates):
                dist = self.xy_distance(p_pos, s_pos)
                cost_matrix[i, j] = dist
                rospy.loginfo(
                    '  Check %s[%d]-Side[%d] dist: %.3fm',
                    primary_label,
                    i,
                    j,
                    dist
                )

        row_ind, col_ind = self.solve_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            dist = float(cost_matrix[i, j])

            if dist <= threshold:
                matches.append({
                    'primary_idx': int(i),
                    'side_idx': int(j),
                    'dist': dist
                })
                rospy.loginfo(
                    '  => Match %s[%d] <-> Side[%d] dist=%.3fm',
                    primary_label,
                    i,
                    j,
                    dist
                )
            else:
                rospy.logwarn(
                    '  => Reject %s[%d] <-> Side[%d] dist=%.3fm > %.2fm',
                    primary_label,
                    i,
                    j,
                    dist,
                    threshold
                )

        return matches

    def build_side_dict(self, matches):
        """side_idx をキーにして match を取り出せるようにする。"""
        result = {}
        for m in matches:
            side_idx = m['side_idx']
            # 通常 Hungarian の結果では side_idx は重複しないが、念のため距離が小さい方を残す。
            if side_idx not in result or m['dist'] < result[side_idx]['dist']:
                result[side_idx] = m
        return result

    # ============================================================
    # 3カメラ統合
    # ============================================================
    def try_integrate(self, force=False):
        """
        3カメラ統合。

        基本方針:
        1. Top-Side をハンガリアン法で対応付ける。
        2. Third-Side もハンガリアン法で対応付ける。
        3. 同じ Side に対して Top と Third が一致していれば TopのXY + SideのZ を採用。
        4. Top と Third が矛盾する場合は、近接俯瞰の誤認識の可能性があるため ThirdのXY + SideのZ を採用。
        5. Top が無く Third だけ合う場合も ThirdのXY + SideのZ を採用。
        """
        if not self.run_detection or self.integration_done:
            return

        if not force:
            # 全カメラの候補が揃っていれば早めに統合してよい。
            all_views_have_candidates = (
                len(self.top_candidates) > 0 and
                len(self.side_candidates) > 0 and
                len(self.third_candidates) > 0
            )

            if not all_views_have_candidates:
                # まだ timeout 前なら、他カメラの結果を待つ。
                if self.integration_started_at is not None:
                    elapsed = (rospy.Time.now() - self.integration_started_at).to_sec()
                    if elapsed < self.integration_timeout:
                        return

        if not self.side_candidates:
            rospy.logwarn_throttle(
                1.0,
                'Cannot integrate: side_candidates is empty. Side camera is needed for Z.'
            )
            return

        if not self.top_candidates and not self.third_candidates:
            rospy.logwarn_throttle(
                1.0,
                'Cannot integrate: both top_candidates and third_candidates are empty.'
            )
            return

        rospy.loginfo(
            '=== 3Cam Integration Start: Top(%d), Side(%d), Third(%d) ===',
            len(self.top_candidates),
            len(self.side_candidates),
            len(self.third_candidates)
        )

        top_side_matches = self.hungarian_match(
            self.top_candidates,
            self.side_candidates,
            self.top_side_threshold,
            'Top'
        )

        third_side_matches = self.hungarian_match(
            self.third_candidates,
            self.side_candidates,
            self.third_side_threshold,
            'Third'
        )

        top_by_side = self.build_side_dict(top_side_matches)
        third_by_side = self.build_side_dict(third_side_matches)

        final_targets = []
        used_side_indices = set()

        # side を基準にしてターゲットを決める。
        # Zは side、XYは top または third から採用する。
        candidate_side_indices = sorted(set(top_by_side.keys()) | set(third_by_side.keys()))

        for side_idx in candidate_side_indices:
            if side_idx in used_side_indices:
                continue

            s_pos = self.side_candidates[side_idx]
            top_match = top_by_side.get(side_idx, None)
            third_match = third_by_side.get(side_idx, None)

            selected_xy = None
            selected_source = None

            # --------------------------------------------------
            # Case 1: Top と Third の両方が同じ Side に合っている
            # --------------------------------------------------
            if top_match is not None and third_match is not None:
                t_pos = self.top_candidates[top_match['primary_idx']]
                th_pos = self.third_candidates[third_match['primary_idx']]
                top_third_dist = self.xy_distance(t_pos, th_pos)

                rospy.loginfo(
                    '  Validate Side[%d]: Top[%d] vs Third[%d] dist=%.3fm',
                    side_idx,
                    top_match['primary_idx'],
                    third_match['primary_idx'],
                    top_third_dist
                )

                if top_third_dist <= self.top_third_threshold:
                    # 近接俯瞰と全体俯瞰が一致 → 精度の高い top のXYを使う
                    selected_xy = (t_pos[0], t_pos[1])
                    selected_source = 'Top confirmed by Third'
                else:
                    # 近接俯瞰と全体俯瞰が矛盾
                    # top が誤認識している可能性があるため、基本は third を使う
                    if self.prefer_third_on_conflict:
                        selected_xy = (th_pos[0], th_pos[1])
                        selected_source = 'Third selected because Top conflicts'
                        rospy.logwarn(
                            '  Conflict at Side[%d]: Top and Third disagree. Use Third XY.',
                            side_idx
                        )
                    else:
                        selected_xy = (t_pos[0], t_pos[1])
                        selected_source = 'Top selected despite conflict'
                        rospy.logwarn(
                            '  Conflict at Side[%d]: Top and Third disagree. Use Top XY by parameter.',
                            side_idx
                        )

            # --------------------------------------------------
            # Case 2: Top だけが Side と合っている
            # --------------------------------------------------
            elif top_match is not None:
                t_pos = self.top_candidates[top_match['primary_idx']]

                if len(self.third_candidates) == 0:
                    # third で検出が無いなら検証できないので top を採用
                    selected_xy = (t_pos[0], t_pos[1])
                    selected_source = 'Top only (no Third candidates)'
                else:
                    # third は何か見えているのに、その side に合う third が無い。
                    # topの誤認識を疑い、かなり近い場合だけ採用する。
                    if top_match['dist'] <= self.strict_top_threshold:
                        selected_xy = (t_pos[0], t_pos[1])
                        selected_source = 'Top only but very close to Side'
                        rospy.logwarn(
                            '  Top[%d]-Side[%d] accepted without Third because dist=%.3fm <= %.2fm',
                            top_match['primary_idx'],
                            side_idx,
                            top_match['dist'],
                            self.strict_top_threshold
                        )
                    else:
                        rospy.logwarn(
                            '  Reject Top[%d]-Side[%d]: not confirmed by Third and dist=%.3fm > strict %.2fm',
                            top_match['primary_idx'],
                            side_idx,
                            top_match['dist'],
                            self.strict_top_threshold
                        )

            # --------------------------------------------------
            # Case 3: Third だけが Side と合っている
            # --------------------------------------------------
            elif third_match is not None:
                th_pos = self.third_candidates[third_match['primary_idx']]
                selected_xy = (th_pos[0], th_pos[1])
                selected_source = 'Third fallback'
                rospy.logwarn(
                    '  Use Third[%d]-Side[%d] fallback because Top is missing or rejected.',
                    third_match['primary_idx'],
                    side_idx
                )

            if selected_xy is not None:
                target = (selected_xy[0], selected_xy[1], s_pos[2])
                final_targets.append(target)
                used_side_indices.add(side_idx)

                rospy.loginfo(
                    '  => Final target Side[%d]: source=%s, base=(x=%.3f, y=%.3f, z=%.3f)',
                    side_idx,
                    selected_source,
                    target[0],
                    target[1],
                    target[2]
                )

        if final_targets:
            final_targets.sort(key=lambda p: p[0])
            self.target_queue = final_targets
            self.run_detection = False
            self.integration_done = True

            rospy.loginfo('SUCCESS: %d weeds ready to send.', len(final_targets))
            self.send_next_target()
        else:
            rospy.logwarn_throttle(
                1.0,
                'FAILED: No valid 3-camera matching result. Check TF, thresholds, and detections.'
            )

    # ============================================================
    # M5へ送信
    # ============================================================
    def send_next_target(self):
        """キューから次の雑草を取り出してM5へ送信。"""
        if not self.target_queue:
            rospy.loginfo('All targets completed.')
            return

        x, y, z = self.target_queue.pop(0)

        # 座標変換: ROS(base_link) -> M5側の座標系
        msg = Float64MultiArray()
        msg.data = [
            -x * 1000.0,
            y * 1000.0,
            -z * 1000.0,
            0.0
        ]

        self.stalk_position_pub.publish(msg)
        rospy.loginfo('Sending target to M5: %s', msg.data[:3])

    def weeding_done_callback(self, msg):
        """M5から除草完了(True)を受け取ったら次を送る。"""
        if msg.data:
            rospy.loginfo('Received done signal from M5.')
            rospy.sleep(1.0)
            self.send_next_target()

    # ============================================================
    # TF変換
    # ============================================================
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
            return (p_out.point.x, p_out.point.y, p_out.point.z)

        except Exception as e:
            rospy.logwarn_throttle(
                1.0,
                'TF transform failed: %s -> %s: %s',
                frame_id,
                self.target_frame,
                e
            )
            return None


if __name__ == '__main__':
    try:
        node = WeedingRecognitionNode3CamHungarian()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
