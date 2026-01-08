from ultralytics import YOLO
import face_recognition
import pickle
import cv2
import numpy as np


class VisionSystem:
    """
    整合 YOLO（小豬偵測） + Face Recognition（人臉識別）

    支援兩種模式：
    - 標準模式 (smart_mode=False): 每幀都處理，適合快速移動場景
    - 智能模式 (smart_mode=True): 動態偵測觸發，TX2 上 FPS 提升 3-5 倍
    """

    def __init__(
        self,
        yolo_weights='best_pig_model.pt',
        face_encoding_file='owner_face.pkl',
        yolo_conf=0.5,
        face_tolerance=0.5,
        smart_mode=False,
        motion_threshold=500
    ):
        """
        初始化視覺系統

        Args:
            yolo_weights: YOLO 權重檔路徑
            face_encoding_file: 主人人臉特徵檔
            yolo_conf: YOLO 置信度閾值 (預設 0.5)
            face_tolerance: 人臉識別閾值 (預設 0.5，越小越嚴格)
            smart_mode: 是否啟用智能優化模式 (預設 False)
            motion_threshold: 動態偵測閾值，僅在 smart_mode=True 時使用 (預設 500)
        """
        mode_text = "智能模式（動態觸發）" if smart_mode else "標準模式"
        print("=" * 50)
        print(f"初始化 Guardian Eye 視覺系統 - {mode_text}")
        print("=" * 50)

        # 載入 YOLO 模型
        print(f"[1/2] 載入 YOLO 模型: {yolo_weights}")
        try:
            self.yolo = YOLO(yolo_weights)
            self.yolo_conf = yolo_conf
            print("      ✅ YOLO 模型載入成功")
        except Exception as e:
            print(f"      ❌ YOLO 模型載入失敗: {e}")
            raise

        # 載入人臉識別器
        print(f"[2/2] 載入人臉識別器: {face_encoding_file}")
        try:
            with open(face_encoding_file, 'rb') as f:
                self.owner_encodings = pickle.load(f)
            self.face_tolerance = face_tolerance
            print(f"      ✅ 已載入 {len(self.owner_encodings)} 個主人特徵")
        except Exception as e:
            print(f"      ❌ 人臉特徵載入失敗: {e}")
            raise

        # 智能模式設定
        self.smart_mode = smart_mode
        self.motion_threshold = motion_threshold
        self.prev_frame = None

        # 快取上一次的結果（智能模式使用）
        self.last_result = {
            'pig_detected': False,
            'pig_confidence': 0.0,
            'pig_bbox': None,
            'person_detected': False,
            'face_result': 'NO_FACE',
            'face_bbox': None,
            'face_confidence': 0.0
        }

        # 效能統計
        self.stats = {
            'total_frames': 0,
            'motion_detected': 0,
            'motion_skipped': 0,
            'yolo_runs': 0,
            'face_recognition_runs': 0
        }

        if smart_mode:
            print(f"      ⚡ 智能模式已啟用（動態閾值: {motion_threshold}）")

        print("=" * 50)
        print("✅ 視覺系統初始化完成！")
        print("=" * 50)

    def _detect_motion(self, frame):
        """
        動態偵測（僅在智能模式使用）

        Args:
            frame: OpenCV 影像

        Returns:
            bool: True=有動靜，False=無動靜
        """
        # 轉灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 第一次呼叫，建立背景
        if self.prev_frame is None:
            self.prev_frame = gray
            return True  # 第一幀預設有動靜

        # 計算差異
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 計算變化面積
        motion_pixels = cv2.countNonZero(thresh)

        # 更新背景
        self.prev_frame = gray

        # 判斷是否有顯著變化
        has_motion = motion_pixels > self.motion_threshold

        return has_motion

    def process_frame(self, frame):
        """
        處理單幀影像（核心函數）

        智能模式：只在偵測到動靜時才處理
        標準模式：每幀都處理

        Args:
            frame: OpenCV 影像（numpy.ndarray, BGR格式）

        Returns:
            dict: {
                'pig_detected': True/False,
                'pig_confidence': 0.0-1.0,
                'pig_bbox': [x1, y1, x2, y2] or None,
                'person_detected': True/False,
                'face_result': 'OWNER'/'STRANGER'/'NO_FACE',
                'face_bbox': [x1, y1, x2, y2] or None,
                'face_confidence': 0.0-1.0
            }
        """
        self.stats['total_frames'] += 1

        # ========== 智能模式：動態偵測觸發 ==========
        if self.smart_mode:
            has_motion = self._detect_motion(frame)

            if not has_motion:
                # 沒有動靜，直接返回上次結果
                self.stats['motion_skipped'] += 1
                return self.last_result

            # 有動靜，繼續處理
            self.stats['motion_detected'] += 1

        # ========== 開始處理影像 ==========
        result = {
            'pig_detected': False,
            'pig_confidence': 0.0,
            'pig_bbox': None,
            'person_detected': False,
            'face_result': 'NO_FACE',
            'face_bbox': None,
            'face_confidence': 0.0
        }

        # ========== Part 1: YOLO 物體偵測 ==========
        self.stats['yolo_runs'] += 1
        yolo_results = self.yolo(frame, conf=self.yolo_conf, verbose=False)

        for box in yolo_results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()

            # 類別 0 = pig（因為我們只訓練了一個類別）
            if class_id == 0:
                result['pig_detected'] = True
                result['pig_confidence'] = confidence
                result['pig_bbox'] = bbox
                print(f"🐷 偵測到小豬！置信度：{confidence:.2%} 位置：{bbox}")
                break  # 只處理第一個小豬

        # ========== Part 2: 人臉識別 ==========
        self.stats['face_recognition_runs'] += 1
        # 轉換顏色（OpenCV 是 BGR，face_recognition 要 RGB）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 偵測人臉位置
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) > 0:
            result['person_detected'] = True
            face_loc = face_locations[0]  # 只處理第一張臉

            # 轉換座標格式（top, right, bottom, left → x1, y1, x2, y2）
            top, right, bottom, left = face_loc
            result['face_bbox'] = [left, top, right, bottom]

            # 提取特徵並比對
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_encodings) > 0:
                encoding = face_encodings[0]

                # 與主人特徵比對
                matches = face_recognition.compare_faces(
                    self.owner_encodings,
                    encoding,
                    tolerance=self.face_tolerance
                )

                distances = face_recognition.face_distance(self.owner_encodings, encoding)

                if True in matches:
                    best_idx = np.argmin(distances)
                    confidence = 1 - distances[best_idx]
                    result['face_result'] = 'OWNER'
                    result['face_confidence'] = confidence
                    print(f"👤 識別為主人（相似度：{confidence:.2%}）")
                else:
                    min_distance = np.min(distances)
                    confidence = 1 - min_distance
                    result['face_result'] = 'STRANGER'
                    result['face_confidence'] = confidence
                    print(f"⚠️  識別為陌生人（最高相似度：{confidence:.2%}）")

        # 更新快取（智能模式使用）
        if self.smart_mode:
            self.last_result = result

        return result

    def draw_results(self, frame, result):
        """
        在影像上繪製偵測結果

        Args:
            frame: 原始影像
            result: process_frame() 的返回值

        Returns:
            繪製後的影像
        """
        output = frame.copy()

        # 繪製小豬框
        if result['pig_detected']:
            x1, y1, x2, y2 = result['pig_bbox']
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 紅色
            label = f"Pig {result['pig_confidence']:.2%}"
            cv2.putText(output, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 繪製人臉框
        if result['person_detected']:
            x1, y1, x2, y2 = result['face_bbox']

            # 根據身份選擇顏色
            if result['face_result'] == 'OWNER':
                color = (0, 255, 0)  # 綠色
                label = f"Owner {result['face_confidence']:.2%}"
            else:
                color = (255, 0, 0)  # 藍色
                label = f"Stranger {result['face_confidence']:.2%}"

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output

    def get_stats(self):
        """
        取得效能統計資訊

        Returns:
            dict: 效能統計
        """
        total = self.stats['total_frames']
        if total == 0:
            return self.stats

        skip_ratio = (self.stats['motion_skipped'] / total) * 100 if self.smart_mode else 0
        process_ratio = (self.stats['motion_detected'] / total) * 100 if self.smart_mode else 100

        return {
            **self.stats,
            'skip_ratio': skip_ratio,
            'process_ratio': process_ratio
        }

    def print_stats(self):
        """印出效能統計"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("📊 效能統計")
        print("=" * 60)
        print(f"模式：          {'智能模式' if self.smart_mode else '標準模式'}")
        print(f"總幀數：        {stats['total_frames']}")

        if self.smart_mode:
            print(f"偵測到動靜：    {stats['motion_detected']} ({stats.get('process_ratio', 0):.1f}%)")
            print(f"跳過處理：      {stats['motion_skipped']} ({stats.get('skip_ratio', 0):.1f}%)")

        print(f"YOLO 執行次數： {stats['yolo_runs']}")
        print(f"人臉識別次數：  {stats['face_recognition_runs']}")
        print("=" * 60)

        if self.smart_mode:
            print(f"⚡ 跳過率：{stats.get('skip_ratio', 0):.1f}% (越高越省電)")
            print("=" * 60)


# ========== 測試程式 ==========
if __name__ == "__main__":
    import sys
    import time

    print("\n" + "=" * 60)
    print("Guardian Eye 視覺系統測試")
    print("=" * 60 + "\n")

    # 詢問使用者選擇模式
    print("請選擇測試模式：")
    print("  [1] 標準模式（每幀都處理）")
    print("  [2] 智能模式（動態觸發，TX2 推薦）")
    choice = input("\n請輸入 (1 或 2，預設 1): ").strip() or "1"

    smart_mode = (choice == "2")
    motion_threshold = 500

    if smart_mode:
        threshold_input = input(f"動態閾值 (預設 {motion_threshold}): ").strip()
        if threshold_input:
            motion_threshold = int(threshold_input)

    # 初始化系統
    vision = VisionSystem(
        yolo_weights='best_pig_model.pt',
        face_encoding_file='owner_face.pkl',
        yolo_conf=0.5,
        face_tolerance=0.5,
        smart_mode=smart_mode,
        motion_threshold=motion_threshold
    )

    print("\n開啟攝影機...")
    print("操作說明：")
    print("  - 按 'q' 退出")
    print("  - 按 's' 顯示統計資訊")
    print("  - 按 'c' 儲存當前畫面")
    print("=" * 60 + "\n")

    # 開啟攝影機（降低解析度以提升效能）
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 無法開啟攝影機！")
        print("提示：")
        print("  1. 確認攝影機已連接")
        print("  2. 嘗試改用 cap = cv2.VideoCapture(1)")
        exit()

    # FPS 計算
    fps_start_time = time.time()
    fps_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法讀取影像")
            break

        fps_frame_count += 1

        # 處理當前幀
        result = vision.process_frame(frame)

        # 繪製結果
        output = vision.draw_results(frame, result)

        # 計算 FPS
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 0:
            fps = fps_frame_count / elapsed_time
        else:
            fps = 0

        # 顯示效能資訊
        stats = vision.get_stats()

        info_line1 = f"FPS: {fps:.1f} | Frames: {stats['total_frames']}"
        cv2.putText(output, info_line1, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if smart_mode:
            skip_ratio = stats.get('skip_ratio', 0)
            info_line2 = f"Processed: {stats['motion_detected']} | Skipped: {stats['motion_skipped']} ({skip_ratio:.1f}%)"
            cv2.putText(output, info_line2, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 顯示
        cv2.imshow('Guardian Eye Vision System', output)

        # 按鍵處理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n使用者退出")
            break
        elif key == ord('s'):
            vision.print_stats()
        elif key == ord('c'):
            filename = f"capture_{stats['total_frames']}.jpg"
            cv2.imwrite(filename, output)
            print(f"📸 已儲存: {filename}")

    # 最終統計
    vision.print_stats()

    cap.release()
    cv2.destroyAllWindows()
    print("\n視覺系統已關閉")
