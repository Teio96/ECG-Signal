import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import statistics
import time
import threading
import queue
import logging
from collections import deque

# ————— 静态报警阈值配置 —————
ALERT_THRESHOLDS = {
    'bradycardia': 60,        # 心动过缓阈值 (BPM)
    'tachycardia': 100,       # 心动过速阈值 (BPM)
    'p_amplitude_low': 0.04,  # P波幅度静态过低阈值 (mV)
    'p_amplitude_high': 0.3,  # P波幅度静态过高阈值 (mV)
    't_amplitude_low': 0.05,  # T波幅度静态过低阈值 (mV)
    't_amplitude_high': 0.5,  # T波幅度静态过高阈值 (mV)
    'baseline_drift': 0.2,    # 基线漂移阈值 (mV)
    'rr_irregularity': 0.15   # RR间期不规则性阈值 (标准差/平均值)
}

# ————— 动态阈值与正常模式判定参数 —————
DYNAMIC_WINDOW_SIZE    = 100   # P/T 波幅度滑动窗口大小
DYNAMIC_INIT_COUNT     = 20    # 滑动窗口至少累积点数后开始更新动态阈值
NORMAL_HR_STD_THRESH   = 3.0   # 心率标准差阈值 (BPM)，低于则视为稳定
NORMAL_PT_MEAN_LOW     = 0.2   # P/T 比典型下限
NORMAL_PT_MEAN_HIGH    = 0.8   # P/T 比典型上限

# ————— 报警确认与去重参数 —————
ALERT_CONFIRM_THRESHOLD = 3     # 连续触发次数达到后才真正报警
ALERT_COOLDOWN          = 5.0   # 同一类型报警最短间隔 (秒)


class ECG_Analyzer:
    """
    心电信号在线分析与报警模块 (优化版)：
      1. 动态阈值：P/T 波幅度根据滑动窗口的均值±1.5×标准差计算
      2. 仅在“正常测量模式”下启用报警：心率与 P/T 比稳定后进入 normal_mode
      3. 多次确认与去重：同一类型报警需连续超过3次才触发，且间隔 <5秒忽略
      4. 暂停分析时不输出缓存报警
    """

    def __init__(self, alert_callback=None, buffer_size=50):
        """
        alert_callback(alert_dict) 被调用时，传入字典:
          {
            'timestamp': 'YYYY-MM-DD HH:MM:SS',
            'type': '报警类型',
            'message': '详细说明'
          }
        """
        self.alert_callback = alert_callback
        self.buffer_size     = buffer_size

        # —— 数据缓冲区 ——
        self.data_buffer     = []            # 原始 data_row 缓存
        self.heart_rates     = []            # 累积心率列表
        self.p_amplitudes    = []            # 累积 P 波幅度
        self.t_amplitudes    = []            # 累积 T 波幅度
        self.baseline_values = []            # 累积基线电压
        self.rr_intervals    = []            # 累积 RR 间期
        self.last_r_time     = None          # 上一个 R 波时间
        self.alert_history   = []            # 所有真正触发的报警

        # —— 动态阈值滑动窗口 ——
        self.p_window = deque(maxlen=DYNAMIC_WINDOW_SIZE)  # 最近 P 波幅度
        self.t_window = deque(maxlen=DYNAMIC_WINDOW_SIZE)  # 最近 T 波幅度
        # 初始动态阈值即为静态阈值
        self.dynamic_p_low  = ALERT_THRESHOLDS['p_amplitude_low']
        self.dynamic_p_high = ALERT_THRESHOLDS['p_amplitude_high']
        self.dynamic_t_low  = ALERT_THRESHOLDS['t_amplitude_low']
        self.dynamic_t_high = ALERT_THRESHOLDS['t_amplitude_high']

        # —— 正常测量模式判定 ——
        self.normal_mode       = False                     # 是否已进入正常测量模式
        self.hr_window         = deque(maxlen=DYNAMIC_WINDOW_SIZE)    # 最近心率值
        self.pt_ratio_window   = deque(maxlen=DYNAMIC_WINDOW_SIZE)    # 最近 P/T 幅度比

        # —— 报警计数与去重 ——
        self.alert_counts    = {
            '心动过缓': 0, '心动过速': 0,
            'P波过低': 0, 'P波过高': 0,
            'T波过低': 0, 'T波过高': 0,
            '基线漂移': 0, '心律不齐': 0
        }
        self.last_alert_time = {key: 0.0 for key in self.alert_counts}  # 记录各类型报警的最后触发时刻
        self.alert_confirm_threshold = ALERT_CONFIRM_THRESHOLD
        self.alert_cooldown          = ALERT_COOLDOWN

        # —— 控制线程与队列 ——
        self.analysis_queue  = queue.Queue()
        self.running         = False
        self.paused          = False
        self.pause_lock      = threading.Lock()
        self.log_lock        = threading.Lock()

        # —— 日志与回调线程 ——
        self._init_logger()
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()

    def _init_logger(self):
        """配置 logging（输出到控制台），并创建报警日志文件 header"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _write_log(self, alert):
        """将报警写入本地日志文件 (CSV)，防止重复写入需加锁"""
        # 这里可以自行修改成写 CSV 或数据库；示例只写到控制台
        pass  # 如需文件保存，可在此处实现

    def pause(self):
        """暂停分析；暂停期间不会调用 alert_callback 也不会 print 新报警"""
        with self.pause_lock:
            self.paused = True

    def resume(self):
        """恢复分析并允许触发报警回调"""
        with self.pause_lock:
            self.paused = False

    def start(self):
        """启动分析器"""
        self.running = True
        logging.info("ECG 分析器已启动")

    def stop(self):
        """停止分析器"""
        self.running = False
        logging.info("ECG 分析器已停止")

    def add_data(self, data_row):
        """
        向分析队列添加一行新数据：
          data_row 格式: [ID, P波位置, P波幅度, QRS区间, T波位置, T波幅度, 心率(BPM), 基线电压]
        """
        if not self.running:
            return
        try:
            self.analysis_queue.put(data_row)
        except Exception as e:
            print(f"添加数据到队列失败: {str(e)}")

    def _analysis_worker(self):
        """后台线程：从队列取数据，依次 _process_data_row & _analyze_current_state"""
        while True:
            try:
                data_row = self.analysis_queue.get()
                if data_row is None:
                    break

                # 暂停时不处理，但也不能执行报警
                with self.pause_lock:
                    if self.paused:
                        continue

                self._process_data_row(data_row)
                self._analyze_current_state()

                self.analysis_queue.task_done()
            except Exception as e:
                print(f"分析线程出错: {str(e)}")

    def _process_data_row(self, data_row):
        """
        处理单行数据，提取关键指标并进行滑窗更新：
          1. 验证并解析 p_amplitude, t_amplitude, heart_rate, baseline, QRS 区间 → RR 间期
          2. 保持 data_buffer、heart_rates、p_amplitudes、t_amplitudes、baseline_values、rr_intervals 的最大长度
          3. 更新滑动窗口：
             - p_window, t_window 用于动态阈值
             - hr_window, pt_ratio_window 用于“正常测量模式”判定
        """
        try:
            # —— 1. 验证数据行有效性 ——
            if (not data_row) or (len(data_row) < 8):
                print(f"无效数据行: {data_row}")
                return
            if not all([data_row[0], data_row[2], data_row[5], data_row[6], data_row[7]]):
                print(f"数据行缺少关键字段: {data_row}")
                return

            data_id = data_row[0]

            # — 解析 p_amplitude —
            try:
                p_amplitude = float(data_row[2])
            except (ValueError, TypeError):
                p_amplitude = 0.0
                print(f"P波幅度转换失败: {data_row[2]}")

            # — 解析 t_amplitude —
            try:
                t_amplitude = float(data_row[5])
            except (ValueError, TypeError):
                t_amplitude = 0.0
                print(f"T波幅度转换失败: {data_row[5]}")

            # — 解析 heart_rate —
            try:
                heart_rate = float(data_row[6])
            except (ValueError, TypeError):
                heart_rate = 0.0
                print(f"心率转换失败: {data_row[6]}")

            # — 解析 baseline —
            try:
                baseline = float(data_row[7])
            except (ValueError, TypeError):
                baseline = 0.0
                print(f"基线电压转换失败: {data_row[7]}")

            # —— 提取 QRS 区间并计算 RR 间期 ——
            qrs_interval = data_row[3] if len(data_row) > 3 else "0-0"
            qrs_start, qrs_end = 0.0, 0.0
            try:
                if '-' in qrs_interval:
                    qrs_start, qrs_end = map(float, qrs_interval.split('-'))
                else:
                    # 若无法解析，则默认长度 0.1 秒
                    qrs_end = qrs_start + 0.1
                    print(f"无法解析 QRS 区间: {qrs_interval}")
            except Exception as e:
                qrs_end = qrs_start + 0.1
                print(f"解析 QRS 区间失败: {str(e)}")

            current_r_time = (qrs_start + qrs_end) / 2.0
            if (self.last_r_time is not None) and (current_r_time > self.last_r_time):
                rr_interval = current_r_time - self.last_r_time
                self.rr_intervals.append(rr_interval)
                if len(self.rr_intervals) > self.buffer_size:
                    self.rr_intervals.pop(0)
            self.last_r_time = current_r_time

            # —— 2. 缓存到主缓冲区 ——
            self.data_buffer.append(data_row)
            self.heart_rates.append(heart_rate)
            self.p_amplitudes.append(p_amplitude)
            self.t_amplitudes.append(t_amplitude)
            self.baseline_values.append(baseline)

            # 保持各缓冲区最大长度
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer.pop(0)
                self.heart_rates.pop(0)
                self.p_amplitudes.pop(0)
                self.t_amplitudes.pop(0)
                self.baseline_values.pop(0)

            # —— 3. 更新 “动态阈值” 滑窗 ——
            self.p_window.append(p_amplitude)
            self.t_window.append(t_amplitude)
            if len(self.p_window) >= DYNAMIC_INIT_COUNT:
                mean_p = statistics.mean(self.p_window)
                std_p  = statistics.stdev(self.p_window)
                self.dynamic_p_low  = max(ALERT_THRESHOLDS['p_amplitude_low'],  mean_p - 1.5 * std_p)
                self.dynamic_p_high = min(ALERT_THRESHOLDS['p_amplitude_high'], mean_p + 1.5 * std_p)
            if len(self.t_window) >= DYNAMIC_INIT_COUNT:
                mean_t = statistics.mean(self.t_window)
                std_t  = statistics.stdev(self.t_window)
                self.dynamic_t_low  = max(ALERT_THRESHOLDS['t_amplitude_low'],  mean_t - 1.5 * std_t)
                self.dynamic_t_high = min(ALERT_THRESHOLDS['t_amplitude_high'], mean_t + 1.5 * std_t)

            # —— 4. 更新 “正常测量模式” 滑窗 ——
            if heart_rate > 0.0:
                self.hr_window.append(heart_rate)
            if (p_amplitude > 0.0) and (t_amplitude > 0.0):
                self.pt_ratio_window.append(p_amplitude / (t_amplitude + 1e-6))

            # —— 5. 判定是否进入正常测量模式 ——
            if not self.normal_mode:
                if (len(self.hr_window) >= DYNAMIC_INIT_COUNT) and (len(self.pt_ratio_window) >= DYNAMIC_INIT_COUNT):
                    hr_std  = statistics.pstdev(self.hr_window)
                    pt_mean = statistics.mean(self.pt_ratio_window)
                    if (hr_std < NORMAL_HR_STD_THRESH) and (NORMAL_PT_MEAN_LOW < pt_mean < NORMAL_PT_MEAN_HIGH):
                        self.normal_mode = True
                        logging.info("【MODE】已进入正常测量模式，开始启用报警功能")

            print(f"处理数据: ID={data_id}, HR={heart_rate}, P={p_amplitude}, T={t_amplitude}")
        except Exception as e:
            print(f"处理数据行失败: {str(e)}\n行内容: {data_row}")

    def _analyze_current_state(self):
        """
        基于缓存的心电数据进行“异常检测”：
          1. 如果未进入 normal_mode，则跳过所有报警
          2. 心动过缓/过速，P/T 波异常 (动态阈值)，基线漂移，RR 不规则性
          3. 所有 _trigger_alert 均为多次确认 + 去重逻辑
        """
        # 至少要有 5 条数据才能做趋势计算
        if len(self.data_buffer) < 5:
            return

        # 仅在“正常测量模式”下才启动报警逻辑
        if not self.normal_mode:
            return

        try:
            # — 心率分析 —
            current_hr = self.heart_rates[-1]
            if current_hr < ALERT_THRESHOLDS['bradycardia']:
                self._trigger_alert(
                    '心动过缓',
                    f"心率过低: {current_hr:.1f} BPM < {ALERT_THRESHOLDS['bradycardia']} BPM"
                )
            elif current_hr > ALERT_THRESHOLDS['tachycardia']:
                self._trigger_alert(
                    '心动过速',
                    f"心率过高: {current_hr:.1f} BPM > {ALERT_THRESHOLDS['tachycardia']} BPM"
                )

            # — P 波异常 (动态阈值) —
            current_p = self.p_amplitudes[-1]
            if len(self.p_window) >= DYNAMIC_INIT_COUNT:
                if current_p < self.dynamic_p_low:
                    self._trigger_alert(
                        'P波过低',
                        f"P波幅度 {current_p:.3f} mV < 动态下限 {self.dynamic_p_low:.3f} mV"
                    )
                elif current_p > self.dynamic_p_high:
                    self._trigger_alert(
                        'P波过高',
                        f"P波幅度 {current_p:.3f} mV > 动态上限 {self.dynamic_p_high:.3f} mV"
                    )

            # — T 波异常 (动态阈值) —
            current_t = self.t_amplitudes[-1]
            if len(self.t_window) >= DYNAMIC_INIT_COUNT:
                if current_t < self.dynamic_t_low:
                    self._trigger_alert(
                        'T波过低',
                        f"T波幅度 {current_t:.3f} mV < 动态下限 {self.dynamic_t_low:.3f} mV"
                    )
                elif current_t > self.dynamic_t_high:
                    self._trigger_alert(
                        'T波过高',
                        f"T波幅度 {current_t:.3f} mV > 动态上限 {self.dynamic_t_high:.3f} mV"
                    )

            # — 基线漂移异常 —
            baseline_drift = max(self.baseline_values) - min(self.baseline_values)
            if baseline_drift > ALERT_THRESHOLDS['baseline_drift']:
                self._trigger_alert(
                    '基线漂移',
                    f"基线漂移过大: {baseline_drift:.3f} mV > {ALERT_THRESHOLDS['baseline_drift']} mV"
                )

            # — RR 间期不规则性 —
            if len(self.rr_intervals) >= 5:
                rr_mean = statistics.mean(self.rr_intervals)
                rr_std  = statistics.stdev(self.rr_intervals)
                irregularity_index = rr_std / rr_mean if rr_mean > 0 else 0.0
                if irregularity_index > ALERT_THRESHOLDS['rr_irregularity']:
                    self._trigger_alert(
                        '心律不齐',
                        f"RR间期不规则性: {irregularity_index:.3f} > {ALERT_THRESHOLDS['rr_irregularity']}"
                    )
        except Exception as e:
            print(f"分析ECG状态时出错: {str(e)}")

    def _trigger_alert(self, alert_type, message):
        """
        真正触发报警前做“多次确认与去重”：
          1. 如果距离上次同类型报警 < alert_cooldown，则直接忽略
          2. 否则连续计数 +1
             - 若计数 < alert_confirm_threshold，则暂不发送
             - 若计数 >= alert_confirm_threshold，才认为有效报警
               • 重置该类型计数为0，并更新 last_alert_time
               • 构建报警字典，加入 alert_history 并调用回调
        """
        now_ts = time.time()
        last_ts = self.last_alert_time.get(alert_type, 0.0)

        # 同类型报警冷却未到，忽略
        if now_ts - last_ts < self.alert_cooldown:
            return

        # 累加连续触发次数
        self.alert_counts[alert_type] += 1
        if self.alert_counts[alert_type] < self.alert_confirm_threshold:
            return

        # 到达确认阈值 → 真正触发
        self.alert_counts[alert_type] = 0
        self.last_alert_time[alert_type] = now_ts

        local_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        alert = {'timestamp': local_ts, 'type': alert_type, 'message': message}
        self.alert_history.append(alert)

        # 如果未暂停，才执行回调与打印
        with self.pause_lock:
            if not self.paused and self.alert_callback:
                self.alert_callback(alert)
                logging.info(f"[ALERT] {alert_type} | {message}")
            elif not self.paused:
                print("[ANALYZER ALERT]", alert_type, message)

    def get_recent_alerts(self, count=5):
        """获取最近的 count 条报警记录"""
        return self.alert_history[-count:]

    def get_summary_statistics(self):
        """获取 ECG 数据的统计摘要"""
        if not self.data_buffer:
            return {}

        return {
            'heart_rate': {
                'current': self.heart_rates[-1] if self.heart_rates else 0,
                'avg_5': statistics.mean(self.heart_rates[-5:]) if len(self.heart_rates) >= 5 else 0,
                'min': min(self.heart_rates) if self.heart_rates else 0,
                'max': max(self.heart_rates) if self.heart_rates else 0
            },
            'p_amplitude': {
                'current': self.p_amplitudes[-1] if self.p_amplitudes else 0,
                'avg_5': statistics.mean(self.p_amplitudes[-5:]) if len(self.p_amplitudes) >= 5 else 0
            },
            't_amplitude': {
                'current': self.t_amplitudes[-1] if self.t_amplitudes else 0,
                'avg_5': statistics.mean(self.t_amplitudes[-5:]) if len(self.t_amplitudes) >= 5 else 0
            },
            'baseline': {
                'current': self.baseline_values[-1] if self.baseline_values else 0,
                'drift': max(self.baseline_values) - min(self.baseline_values) if self.baseline_values else 0
            },
            'rr_irregularity': self._calculate_rr_irregularity()
        }

    def _calculate_rr_irregularity(self):
        """计算 RR 间期不规则性指数"""
        if len(self.rr_intervals) < 5:
            return 0
        rr_mean = statistics.mean(self.rr_intervals)
        rr_std  = statistics.stdev(self.rr_intervals)
        return rr_std / rr_mean if rr_mean > 0 else 0

    def plot_trends(self, save_path=None):
        """
        绘制 ECG 趋势图（与原接口一致）：
          • 心率、P/T 波幅度、基线漂移、RR 间期趋势
        """
        if len(self.data_buffer) < 10:
            print("数据不足，无法绘制趋势图")
            return

        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle('ECG 趋势分析')

        time_points = list(range(len(self.heart_rates)))

        # — 心率趋势 —
        axs[0].plot(time_points, self.heart_rates, 'b-', label='心率')
        axs[0].axhline(y=ALERT_THRESHOLDS['bradycardia'], color='r', linestyle='--', label='心动过缓阈值')
        axs[0].axhline(y=ALERT_THRESHOLDS['tachycardia'], color='r', linestyle='--', label='心动过速阈值')
        axs[0].set_ylabel('心率 (BPM)')
        axs[0].set_title('心率趋势')
        axs[0].legend()
        axs[0].grid(True)

        # — P/T 波幅度趋势 —
        axs[1].plot(time_points, self.p_amplitudes, 'g-', label='P波幅度')
        axs[1].plot(time_points, self.t_amplitudes, 'm-', label='T波幅度')
        axs[1].axhline(y=ALERT_THRESHOLDS['p_amplitude_low'], color='r', linestyle='--', label='P波低阈值')
        axs[1].axhline(y=ALERT_THRESHOLDS['p_amplitude_high'], color='r', linestyle='--', label='P波高阈值')
        axs[1].axhline(y=ALERT_THRESHOLDS['t_amplitude_low'], color='c', linestyle='--', label='T波低阈值')
        axs[1].axhline(y=ALERT_THRESHOLDS['t_amplitude_high'], color='c', linestyle='--', label='T波高阈值')
        axs[1].set_ylabel('幅度 (mV)')
        axs[1].set_title('P波和T波幅度趋势')
        axs[1].legend()
        axs[1].grid(True)

        # — 基线漂移趋势 —
        axs[2].plot(time_points, self.baseline_values, 'k-', label='基线电压')
        axs[2].axhline(y=ALERT_THRESHOLDS['baseline_drift'] / 2, color='r', linestyle='--', label='漂移阈值')
        axs[2].axhline(y=-ALERT_THRESHOLDS['baseline_drift'] / 2, color='r', linestyle='--')
        axs[2].set_ylabel('电压 (mV)')
        axs[2].set_title('基线漂移趋势')
        axs[2].legend()
        axs[2].grid(True)

        # — RR 间期趋势 —
        if len(self.rr_intervals) > 5:
            rr_points = list(range(len(self.rr_intervals)))
            axs[3].plot(rr_points, self.rr_intervals, 'b-', label='RR间期')
            axs[3].set_xlabel('时间点')
            axs[3].set_ylabel('间期 (秒)')
            axs[3].set_title('RR间期趋势')
            axs[3].legend()
            axs[3].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"趋势图已保存到: {save_path}")
        else:
            plt.show()


# ———— 全局实例与外部接口 ————
global_analyzer = ECG_Analyzer()

def analyze_data(data_row):
    """兼容旧接口：直接将数据行传给全局分析器"""
    global_analyzer.add_data(data_row)

def start_analysis(alert_callback=None):
    """启动 ECG 分析：生成新的全局分析器并调用 start()"""
    global global_analyzer
    global_analyzer = ECG_Analyzer(alert_callback)
    global_analyzer.start()

def stop_analysis():
    """停止 ECG 分析"""
    global_analyzer.stop()

def get_recent_alerts(count=5):
    """获取最近 count 条报警记录"""
    return global_analyzer.get_recent_alerts(count)

def get_summary_statistics():
    """获取 ECG 数据统计摘要"""
    return global_analyzer.get_summary_statistics()

def plot_trends(save_path=None):
    """绘制 ECG 趋势图"""
    global_analyzer.plot_trends(save_path)
