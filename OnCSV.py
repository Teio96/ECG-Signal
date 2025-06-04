import numpy as np
import pandas as pd
import time


class CSVDataCollector:
    """从CSV文件模拟实时数据采集"""

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.ch_label = None
        self.sampling_rate = None
        self.all_voltage_data = np.array([])
        self.total_points = 0
        self._load_csv()

    def _load_csv(self):
        """
        读取CSV文件，格式要求：
        第一行：导联名称（字符串）
        第二行：采样率（数字，单位Hz）
        第三行及以后：各采样点的电压值（单位mV），在第一列
        """
        try:
            df = pd.read_csv(self.csv_file, header=None)
            if df.shape[0] < 3:
                raise ValueError("CSV 文件行数不足，需至少包含导联名称、采样率和一个电压值。")

            # 第一行：导联名称
            self.ch_label = str(df.iat[0, 0])
            # 第二行：采样率
            self.sampling_rate = float(df.iat[1, 0])
            # 第三行及以后：电压数据，取第一列
            voltage_series = df.iloc[2:, 0]
            self.all_voltage_data = voltage_series.astype(float).values
            self.total_points = len(self.all_voltage_data)

            print(f"已读取导联: {self.ch_label}")
            print(f"成功读取 {self.total_points} 个数据点，采样率: {self.sampling_rate} Hz")
        except Exception as e:
            print(f"CSV 加载错误: {e}")
            self.all_voltage_data = np.array([])
            self.total_points = 0
            self.sampling_rate = 0

    def start_stream(self, data_queue):
        """
        启动数据流模拟，将 (timestamp, data_chunk) 放入 data_queue。
        如果无数据或采样率非法，则直接发送结束信号。
        """
        if self.total_points == 0 or self.sampling_rate <= 0:
            print("无可用数据或采样率非法，start_stream 退出")
            data_queue.put((None, None))
            return

        self.running = True
        self.index = 0
        self.start_time = time.time()

        # 调试信息：确认加载情况
        print(f"[CSVDataCollector] start_stream: total_points={self.total_points}, sampling_rate={self.sampling_rate}")

        while self.running and self.index < self.total_points:
            elapsed = time.time() - self.start_time
            expected_index = min(int(elapsed * self.sampling_rate), self.total_points)
            if expected_index > self.index:
                points = self.all_voltage_data[self.index:expected_index]
                self.index = expected_index
                packet_time = time.time()
                data_queue.put((packet_time, points))
            time.sleep(0.05)

        # 流结束信号：发送 (None, None)
        data_queue.put((None, None))
        print("[CSVDataCollector] CSV 数据流结束")

    def stop_stream(self):
        """停止数据流"""
        self.running = False
