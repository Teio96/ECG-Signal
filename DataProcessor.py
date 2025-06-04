# DataProcessor.py

import threading
import numpy as np
import time
import collections
import queue
import traceback
import matplotlib
matplotlib.use('TkAgg')  # 设置使用 TkAgg 后端
from scipy.signal import welch, find_peaks, butter, filtfilt
from DataAnalyze import analyze_data

class DataProcessor:
    def __init__(self, data_queue, result_queue, callback, raw_data_callback, sampling_rate):
        """
        data_queue 中的元素现在是 (packet_time, voltage_list)。
        packet_time 是这一包最后一个点的真实 time.time() 时间戳（秒），
        voltage_list 是长度为 batch_size 的电压数据列表（已减去直流偏置并放大三倍）。
        """
        self.data_queue = data_queue
        self.result_queue = result_queue
        self.callback = callback
        self.raw_heartbeat_callback = None
        self.raw_data_callback = raw_data_callback
        self.running = True
        self.data_id = 0
        self.sampling_rate = sampling_rate  # 250 Hz

        # 显示用缓冲区
        self.time_buffer = collections.deque()
        self.voltage_buffer = collections.deque()
        self.current_time = 0.0

        # ECG 分析用变量
        self.r_peaks = []
        self.last_peak = 0
        self.components = {
            'baseline': np.array([]),
            'P': [],
            'QRS': [],
            'T': []
        }

        # 分析缓冲区设置
        self.ANALYSIS_WINDOW_SECONDS = 10
        self.analysis_buffer_size = int(self.ANALYSIS_WINDOW_SECONDS * self.sampling_rate)
        self.time_analysis = collections.deque(maxlen=self.analysis_buffer_size)
        self.voltage_analysis = collections.deque(maxlen=self.analysis_buffer_size)

        # FFT 分析器
        self.fft_analyzer = FFTAnalyzer(sampling_rate)
        self.fft_callback = None

        # 心跳状态跟踪（重构版）
        self.heartbeat_states = {}  # {r_peak_time: heartbeat_state}
        self.last_completed_heartbeat = None
        self.heartbeat_counter = 0
        self.first_heartbeat = True
        self.min_heartbeat_interval = 0.3  # 秒

        # 可视化回调
        self.visualization_callback = None

        # 已报告 R 波查重
        self.reported_r_peaks = collections.deque(maxlen=50)
        self.duplicate_threshold = 0.3  # 秒

        # 用于输出的递增 ID
        self.output_heartbeat_counter = 0

        # 插值绘图用
        self.dt_target = 1.0 / self.sampling_rate
        self.dt_tol = self.dt_target * 1.5
        self.t_prev = None
        self.v_prev = None

        self.paused = False
        self.pause_event = threading.Event()

    def set_visualization_callback(self, callback):
        """设置可视化回调函数"""
        self.visualization_callback = callback

    def set_raw_data_callback(self, callback):
        """
        设置原始每个采样点的写入 CSV 回调函数。
        回调时会传入 (timestamp, voltage)。
        """
        self.raw_data_callback = callback

    def set_raw_heartbeat_callback(self, callback):
        """设置未经处理的心跳波形回调函数（Platform 将用以写入 CSV）"""
        self.raw_heartbeat_callback = callback

    def detect_r_peaks(self, signal, fs):
        """鲁棒的 R 波检测算法（改进版）"""
        try:
            # 带通滤波 (5-15 Hz)
            nyquist = fs / 2.0
            b, a = butter(1, [5.0 / nyquist, 15.0 / nyquist], btype='bandpass')
            filtered = filtfilt(b, a, signal)

            # 微分
            diff = np.diff(filtered)

            # 平方
            squared = diff ** 2

            # 移动平均积分 (窗口大小约30ms)
            window_size = int(0.03 * fs)
            window = np.ones(window_size) / window_size
            integrated = np.convolve(squared, window, mode='same')

            # 自适应阈值检测
            peaks = []
            threshold = np.max(integrated) * 0.2
            refractory_period = int(0.2 * fs)  # 200ms 不应期
            last_peak = -refractory_period

            for i in range(1, len(integrated) - 1):
                if i - last_peak < refractory_period:
                    continue
                if integrated[i] > integrated[i - 1] and integrated[i] > integrated[i + 1] and integrated[i] > threshold:
                    # 在原始信号中寻找精确的 R 波位置
                    search_start = max(0, i - int(0.05 * fs))
                    search_end = min(len(signal) - 1, i + int(0.05 * fs))
                    r_peak_idx = np.argmax(signal[search_start:search_end]) + search_start

                    peaks.append(r_peak_idx)
                    last_peak = r_peak_idx

                    # 动态更新阈值
                    if len(peaks) > 8:
                        recent_peaks = peaks[-8:]
                        threshold = 0.5 * threshold + 0.5 * (
                            np.mean([integrated[p] for p in recent_peaks]) * 0.4
                        )
            return peaks

        except Exception as e:
            print(f"R 波检测错误: {str(e)}")
            return []

    def detect_and_update_heartbeats(self, voltage_array, time_array, fs):
        """
        检测波形并更新心跳状态 - 综合 R/P/QRS/T 检测与状态管理
        返回 (components_dict, r_peak_times_list)
        """
        # 1. 检测 R 波
        r_peak_indices = self.detect_r_peaks(voltage_array, fs)
        r_peak_times = [time_array[i] for i in r_peak_indices if i < len(time_array)]

        # 初始化组件字典
        components = {'P': [], 'QRS': [], 'T': []}
        detected_t_waves = []

        # 遍历每个 R 波，检测 QRS 范围与 T 波
        for r_idx in r_peak_indices:
            if r_idx >= len(time_array):
                continue
            r_time = time_array[r_idx]

            # QRS 复合波左右 80ms 区间
            qrs_duration = int(0.08 * fs)
            q_start = max(0, r_idx - qrs_duration)
            q_end = min(len(voltage_array) - 1, r_idx + qrs_duration)

            # 查找 Q 点与 S 点
            try:
                q_point = q_start + np.argmin(voltage_array[q_start:r_idx])
                s_point = r_idx + np.argmin(voltage_array[r_idx:q_end])
            except:
                q_point = r_idx
                s_point = r_idx

            # 记录 QRS 时间
            qrs_start_time = time_array[q_point] if q_point < len(time_array) else r_time
            qrs_end_time = time_array[s_point] if s_point < len(time_array) else r_time

            # 检测 T 波 (S 后 200-400ms 范围内的峰)
            t_time = None
            t_amplitude = None
            try:
                t_search_start = min(len(voltage_array) - 1, s_point + int(0.2 * fs))
                t_search_end = min(len(voltage_array) - 1, s_point + int(0.4 * fs))
                if t_search_end > t_search_start:
                    t_segment = voltage_array[t_search_start:t_search_end]
                    if len(t_segment) > 0:
                        t_idx = t_search_start + np.argmax(t_segment)
                        if t_idx < len(time_array):
                            t_time = time_array[t_idx]
                            t_amplitude = voltage_array[t_idx]
                            detected_t_waves.append({
                                't_time': t_time,
                                't_amplitude': t_amplitude,
                                'r_peak_time': r_time,
                                's_point_time': time_array[s_point] if s_point < len(time_array) else r_time
                            })
                            components['T'].append(t_time)
            except Exception as e:
                print(f"T 波检测错误: {e}")

            # P 波检测 (QRS 前 120-200ms)
            p_time = None
            p_amplitude = None
            try:
                p_search_start = max(0, q_point - int(0.2 * fs))
                p_search_end = max(0, q_point - int(0.12 * fs))
                if p_search_end > p_search_start:
                    p_segment = voltage_array[p_search_start:p_search_end]
                    if len(p_segment) > 0:
                        p_idx = p_search_start + np.argmax(p_segment)
                        if p_idx < len(time_array):
                            p_time = time_array[p_idx]
                            p_amplitude = voltage_array[p_idx]
                            components['P'].append(p_time)
            except Exception as e:
                print(f"P 波检测错误: {e}")

            # 基线电压：取 Q 点前 50ms 的平均
            if q_point < len(voltage_array):
                baseline_window = int(0.05 * fs)
                start_idx = max(0, q_point - baseline_window)
                segment = voltage_array[start_idx:q_point]
                baseline_voltage = float(np.mean(segment)) if len(segment) > 0 else float(voltage_array[q_point])
            else:
                baseline_voltage = None

            # 检查此 R 波是新心跳还是与已有状态合并
            is_new = True
            existing_key = None
            for existing_r in list(self.heartbeat_states.keys()):
                if abs(existing_r - r_time) < self.min_heartbeat_interval:
                    is_new = False
                    existing_key = existing_r
                    break

            if is_new:
                # 创建新心跳状态条目
                heartbeat_id = self.heartbeat_counter
                self.heartbeat_counter += 1
                self.heartbeat_states[r_time] = {
                    'id': heartbeat_id,
                    'r_peak_time': r_time,
                    'r_index': r_idx,
                    'p_time': p_time,
                    'p_amplitude': p_amplitude,
                    'qrs_start': qrs_start_time,
                    'qrs_end': qrs_end_time,
                    't_time': None,
                    't_amplitude': None,
                    'baseline_voltage': baseline_voltage,
                    'detected_time': time_array[-1]
                }

            # 关联 T 波到相应的心跳状态
            for t_wave in detected_t_waves:
                if t_wave['r_peak_time'] == r_time or (
                   abs(t_wave['s_point_time'] - r_time) < 0.5 and t_wave['t_time'] > r_time):
                    key = existing_key if not is_new else r_time
                    if key in self.heartbeat_states:
                        self.heartbeat_states[key]['t_time'] = t_wave['t_time']
                        self.heartbeat_states[key]['t_amplitude'] = t_wave['t_amplitude']
                    break

        # 把所有心跳状态中的 QRS 时间写入 components
        for state in self.heartbeat_states.values():
            if state['qrs_start'] is not None and state['qrs_end'] is not None:
                components['QRS'].append((state['qrs_start'], state['qrs_end']))

        return components, r_peak_times

    def detect_ecg_components(self, signal, fs, r_peaks, current_time):
        """
        检测 ECG 成分并关联到具体心跳 - 备用函数，保持原有逻辑。
        """
        try:
            time_array = np.array(self.time_analysis)
            components = {'P': [], 'QRS': [], 'T': []}
            valid_r_peaks = [r for r in r_peaks if r < len(time_array)]

            for r in valid_r_peaks:
                # QRS 区间
                qrs_duration = int(0.08 * fs)
                q_start = max(0, r - qrs_duration)
                q_end = r
                q_point = q_start + np.argmin(signal[q_start:q_end])

                s_start = r
                s_end = min(len(signal) - 1, r + qrs_duration)
                s_point = s_start + np.argmin(signal[s_start:s_end])
                components['QRS'].append((q_point, s_point))

                # P 波 (QRS 前 120-200ms)
                p_start = max(0, q_point - int(0.2 * fs))
                p_end = q_point - int(0.12 * fs)
                if p_end > p_start:
                    p_segment = signal[p_start:p_end]
                    if len(p_segment) > 0:
                        p_idx = p_start + np.argmax(p_segment)
                        components['P'].append(p_idx)

                # T 波 (QRS 后 200-400ms)
                t_start = s_point + int(0.2 * fs)
                t_end = min(len(signal) - 1, s_point + int(0.4 * fs))
                if t_end > t_start:
                    t_segment = signal[t_start:t_end]
                    if len(t_segment) > 0:
                        t_idx = t_start + np.argmax(t_segment)
                        components['T'].append(t_idx)

            return components
        except Exception as e:
            print(f"ECG 成分检测错误: {e}")
            return {'P': [], 'QRS': [], 'T': []}

    def process_buffer(self):
        """对缓冲区数据进行心跳检测、组件提取、回调输出等"""
        if len(self.voltage_buffer) == 0:
            return

        try:
            time_array = np.array(self.time_analysis)
            voltage_array = np.array(self.voltage_analysis)

            # 检测并更新心跳状态
            components, r_peak_times = self.detect_and_update_heartbeats(
                voltage_array, time_array, self.sampling_rate
            )

            self.components = components
            self.r_peaks = r_peak_times

            # 计算心率（基于最后两个 R 峰）
            hr = None
            if len(r_peak_times) >= 2:
                rr_interval = r_peak_times[-1] - r_peak_times[-2]
                if rr_interval > 0:
                    hr = 60.0 / rr_interval

            # 后处理：若某心跳缺少 T 波，则在 QRS 后 0.2-0.4 秒区间再次尝试关联
            current_time = time_array[-1] if len(time_array) > 0 else 0
            for r_time, state in list(self.heartbeat_states.items()):
                if state.get('t_time') is None and state.get('qrs_end') is not None:
                    qrs_end_time = state['qrs_end']
                    t_search_start = qrs_end_time + 0.2
                    t_search_end = qrs_end_time + 0.4
                    possible_t_indices = np.where(
                        (time_array >= t_search_start) &
                        (time_array <= t_search_end) &
                        (voltage_array > np.percentile(voltage_array, 70))
                    )[0]
                    if len(possible_t_indices) > 0:
                        t_idx = possible_t_indices[np.argmax(voltage_array[possible_t_indices])]
                        t_time = time_array[t_idx]
                        t_amp = voltage_array[t_idx]
                        state['t_time'] = t_time
                        state['t_amplitude'] = t_amp

            # 完成并输出过期心跳
            expired = []
            for r_time, state in list(self.heartbeat_states.items()):
                if not isinstance(state, dict):
                    expired.append(r_time)
                    continue
                # 过期条件：距检测已经过 >0.5s 或距 R 峰 >1.0s
                if (current_time - state.get('detected_time', 0) > 0.5) or (current_time - r_time > 1.0):
                    try:
                        self.complete_heartbeat(state)
                    except Exception as e:
                        print(f"完成心跳出错: {e}\n{traceback.format_exc()}")
                    expired.append(r_time)

            for r_time in expired:
                if r_time in self.heartbeat_states:
                    del self.heartbeat_states[r_time]

        except Exception as e:
            print(f"处理缓冲区错误: {e}\n{traceback.format_exc()}")
        finally:
            # 清空绘图缓冲区
            self.time_buffer.clear()
            self.voltage_buffer.clear()

    def stop(self):
        """停止数据处理并清空所有缓冲区"""
        self.running = False
        self.time_buffer.clear()
        self.voltage_buffer.clear()
        self.time_analysis.clear()
        self.voltage_analysis.clear()
        print("DataProcessor: 已停止并清空缓冲区")

    def pause(self):
        """暂停数据处理"""
        self.paused = True
        self.pause_event.clear()

    def resume(self):
        """恢复数据处理"""
        self.paused = False
        self.pause_event.set()

    def process_data(self):
        """
        处理数据的主循环。
        从 data_queue 获取 (packet_time, data_chunk)，
        然后为这一包中的每个采样点分配近似时间并进行插值与处理。
        """
        while self.running:
            if self.paused:
                self.pause_event.wait()
                continue

            try:
                packet_time, data_chunk = self.data_queue.get(timeout=0.1)
                if data_chunk is None:
                    print("DataProcessor: 收到停止信号，结束处理线程")
                    break

                N = len(data_chunk)
                # 构造这一包内每个点的近似时刻
                times = [
                    packet_time - (N - 1 - i) * (1.0 / self.sampling_rate)
                    for i in range(N)
                ]

                # 原始数据回调（写 CSV），传入 (timestamp, voltage)
                if self.raw_data_callback:
                    for t_pt, voltage in zip(times, data_chunk):
                        try:
                            self.raw_data_callback(t_pt, voltage)
                        except Exception as e:
                            print(f"raw_data_callback 调用失败: {e}")

                # 添加整包到 FFT 分析器
                try:
                    self.fft_analyzer.add_data(data_chunk)
                except Exception as e:
                    print(f"添加到 FFT 分析器失败: {e}\n{traceback.format_exc()}")

                # 逐点更新缓冲区与可视化，并做插值
                for t_cur, v_cur in zip(times, data_chunk):
                    # 若有上一个点，检查 dt 是否过大，需要插虚拟点
                    if self.t_prev is not None:
                        dt = t_cur - self.t_prev
                        if dt > self.dt_tol:
                            # 需要插入 N_virtual 个线性点
                            N_virtual = int(round(dt / self.dt_target)) - 1
                            for k in range(1, N_virtual + 1):
                                t_virtual = self.t_prev + k * (dt / (N_virtual + 1))
                                v_virtual = self.v_prev + (v_cur - self.v_prev) * (k / (N_virtual + 1))
                                # 插入虚拟点
                                self.time_buffer.append(t_virtual)
                                self.voltage_buffer.append(v_virtual)
                                self.time_analysis.append(t_virtual)
                                self.voltage_analysis.append(v_virtual)
                                # 可视化回调也用虚拟点填充
                                if self.visualization_callback:
                                    try:
                                        self.visualization_callback(
                                            t_virtual, v_virtual, self.r_peaks, self.components
                                        )
                                    except Exception:
                                        pass

                    # 再加入当前真实点
                    self.current_time = t_cur
                    self.time_buffer.append(t_cur)
                    self.voltage_buffer.append(v_cur)
                    self.time_analysis.append(t_cur)
                    self.voltage_analysis.append(v_cur)

                    if self.visualization_callback:
                        try:
                            self.visualization_callback(
                                t_cur, v_cur, self.r_peaks, self.components
                            )
                        except Exception:
                            pass

                    self.t_prev = t_cur
                    self.v_prev = v_cur

                # 处理缓冲区（检测心跳、生成组件、回调输出）
                self.process_buffer()

                # 更新 FFT 分析结果并回调
                try:
                    fft_result = self.fft_analyzer.update()
                    if fft_result and self.fft_callback:
                        self.fft_callback(fft_result)
                except Exception as e:
                    print(f"更新 FFT 分析时出错: {e}\n{traceback.format_exc()}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"process_data 错误: {e}\n{traceback.format_exc()}")

        print("DataProcessor: 数据处理线程结束")

    def complete_heartbeat(self, state):
        """完成一个心跳周期的处理并输出结果"""
        if self.first_heartbeat:
            self.first_heartbeat = False
            return

        try:
            if not isinstance(state, dict):
                print(f"警告: 无效的心跳状态类型: {type(state)}")
                return

            r_time = state.get('r_peak_time')
            if r_time is None:
                print("警告: 心跳状态缺少 r_peak_time")
                return

            # 查重：若已报告过此 R 峰，则跳过
            for reported_time in self.reported_r_peaks:
                if abs(reported_time - r_time) < self.duplicate_threshold:
                    return
            self.reported_r_peaks.append(r_time)

            # 提取原始心跳波形（前 0.2s ~ 后 0.4s）
            try:
                r_index = state.get('r_index')
                if r_index is not None:
                    pre_len = int(0.2 * self.sampling_rate)
                    post_len = int(0.4 * self.sampling_rate)
                    start_idx = max(0, r_index - pre_len)
                    end_idx = min(len(self.voltage_analysis), r_index + post_len)
                    raw_segment = list(self.voltage_analysis)[start_idx:end_idx]
                    if self.raw_heartbeat_callback:
                        self.raw_heartbeat_callback(raw_segment)
            except Exception as e:
                print(f"提取原始心跳波形失败: {e}")

            # 递增输出心跳计数
            self.output_heartbeat_counter += 1

            # 计算心率（以最后两个 R 峰为准）
            hr_value = None
            if len(self.r_peaks) >= 2:
                try:
                    last_r = self.r_peaks[-1]
                    prev_r = self.r_peaks[-2]
                    rr_interval = last_r - prev_r
                    if rr_interval > 0:
                        hr_value = 60.0 / rr_interval
                except Exception as e:
                    print(f"心率计算错误: {e}")

            # 安全取值函数
            def safe_value(key, fmt="{}"):
                val = state.get(key)
                if val is None:
                    return "N/A"
                try:
                    return fmt.format(val)
                except:
                    return str(val)

            # 构建输出行：[
            #   心跳ID, P波时间, P波幅度-基线, QRS 起止, T波时间,
            #   T波幅度-基线, 心率, 基线电压
            # ]
            row = [
                self.output_heartbeat_counter,
                safe_value('p_time', "{:.3f}"),
                (
                    "{:.3f}".format(
                        state.get('p_amplitude') - state.get('baseline_voltage')
                    )
                    if (state.get('p_amplitude') is not None and state.get('baseline_voltage') is not None)
                    else "N/A"
                ),
                f"{safe_value('qrs_start', '{:.3f}')}-{safe_value('qrs_end', '{:.3f}')}",
                safe_value('t_time', "{:.3f}"),
                (
                    "{:.3f}".format(
                        state.get('t_amplitude') - state.get('baseline_voltage')
                    )
                    if (state.get('t_amplitude') is not None and state.get('baseline_voltage') is not None)
                    else "N/A"
                ),
                f"{hr_value:.1f}" if hr_value is not None else "N/A",
                safe_value('baseline_voltage', "{:.3f}")
            ]

            # 回调输出给 Platform
            if self.callback:
                self.callback(row)

            # 实时分析调用
            try:
                analyze_data(row)
            except Exception as e:
                print(f"实时分析出错: {e}")

        except Exception as e:
            print(f"完成心跳时出错: {e}\n{traceback.format_exc()}")

    def detect_r_peaks_pan_tompkins(self, signal, fs):
        """
        Pan-Tompkins 改进算法，仅做备用示例。
        """
        try:
            nyquist = fs / 2.0
            b, a = butter(1, [5.0 / nyquist, 15.0 / nyquist], btype='bandpass')
            filtered = filtfilt(b, a, signal)

            diff = np.diff(filtered)
            squared = diff ** 2
            window_size = int(0.03 * fs)
            window = np.ones(window_size) / window_size
            integrated = np.convolve(squared, window, mode='same')

            peaks = []
            threshold = np.max(integrated) * 0.2
            refractory_period = int(0.2 * fs)
            last_peak = -refractory_period

            for i in range(1, len(integrated) - 1):
                if i - last_peak < refractory_period:
                    continue
                if integrated[i] > integrated[i - 1] and integrated[i] > integrated[i + 1] and integrated[i] > threshold:
                    search_start = max(0, i - int(0.05 * fs))
                    search_end = min(len(signal) - 1, i + int(0.05 * fs))
                    r_peak = np.argmax(signal[search_start:search_end]) + search_start
                    peaks.append(r_peak)
                    last_peak = r_peak
                    if len(peaks) > 8:
                        recent_peaks = peaks[-8:]
                        threshold = 0.5 * threshold + 0.5 * (
                            np.mean([integrated[p] for p in recent_peaks]) * 0.4
                        )
            return peaks
        except Exception as e:
            print(f"Pan-Tompkins R 波检测错误: {e}")
            return []

class FFTAnalyzer:
    """高效且鲁棒的 FFT 分析器"""
    def __init__(self, sampling_rate):
        self.sampling_rate = float(sampling_rate)
        buffer_size = int(self.sampling_rate * 10)  # 10 秒缓冲区
        self.buffer = collections.deque(maxlen=buffer_size)
        self.last_update = time.time()
        self.update_interval = 1.0  # 每 1 秒更新一次
        self.min_samples = max(256, int(self.sampling_rate * 2))

    def add_data(self, data_points):
        """添加新数据到缓冲区"""
        try:
            if not isinstance(data_points, (list, np.ndarray)):
                if isinstance(data_points, (int, float)):
                    data_points = [data_points]
                else:
                    print(f"FFTAnalyzer: 警告 - 添加了无效数据类型: {type(data_points)}")
                    return
            for pt in data_points:
                if isinstance(pt, (int, float)):
                    self.buffer.append(pt)
                else:
                    print(f"FFTAnalyzer: 警告 - 跳过无效数据点: {pt} (类型={type(pt)})")
        except Exception as e:
            print(f"FFTAnalyzer: 添加数据出错: {e}\n{traceback.format_exc()}")

    def compute(self):
        """计算 Welch PSD 并返回心率估计结果"""
        try:
            if len(self.buffer) < self.min_samples:
                return None
            start_time = time.time()

            signal = np.array(self.buffer)
            signal = signal - np.mean(signal)
            nperseg = min(2048, len(signal))
            noverlap = nperseg // 2

            freqs, psd = welch(
                signal,
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann',
                scaling='density'
            )

            hr_band = (freqs >= 0.8) & (freqs <= 3.0)
            freqs_hr = freqs[hr_band]
            psd_hr = psd[hr_band]

            if len(freqs_hr) == 0:
                return {
                    'freqs': freqs,
                    'psd': psd,
                    'hr': None,
                    'peak_freq': None,
                    'peak_power': None
                }

            median_power = np.median(psd_hr)
            threshold = median_power * 3

            peaks, properties = find_peaks(
                psd_hr,
                height=threshold,
                prominence=0.5,
                width=2
            )
            if len(peaks) == 0:
                peak_idx = np.argmax(psd_hr)
            else:
                peak_idx = peaks[np.argmax(properties['prominences'])]

            hr = freqs_hr[peak_idx] * 60
            result = {
                'freqs': freqs,
                'psd': psd,
                'hr': hr,
                'peak_freq': freqs_hr[peak_idx],
                'peak_power': psd_hr[peak_idx]
            }

            # 清空缓冲区，等待下次积累
            self.buffer.clear()
            elapsed = time.time() - start_time
            return result

        except Exception as e:
            print(f"FFTAnalyzer: 计算错误: {e}")
            return None

    def update(self):
        """定时触发计算并返回结果，如果缓存不足则返回 None"""
        now = time.time()
        if now - self.last_update >= self.update_interval:
            self.last_update = now
            return self.compute()
        return None