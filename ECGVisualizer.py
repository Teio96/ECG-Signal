# ECGVisualizer.py

import numpy as np
import time
import collections
import threading
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
import os
import tkinter as tk

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 防止负号乱码

class ECGVisualizer:
    """ECG 可视化界面——在 Raspberry Pi 上保持 blit=True，
       并在波形显示时对明显空隙做线性插值补齐"""

    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.running = True
        self.data_lock = threading.Lock()

        # 显示窗口长度（秒）——保持 5 秒
        self.DISPLAY_WINDOW_SECONDS = 5
        self.window_size = int(self.DISPLAY_WINDOW_SECONDS * self.sampling_rate)

        # 存放时间、电压数据（固定长度队列）
        self.time_data = collections.deque(maxlen=self.window_size)
        self.voltage_data = collections.deque(maxlen=self.window_size)

        self.current_time = 0.0
        self.r_peaks = []
        self.components = {'P': [], 'QRS': [], 'T': []}
        self.data_started = False  # 仅当第一次收到数据时切换状态

        # 插值补齐用
        self.t_prev = None
        self.v_prev = None
        self.dt_target = 1.0 / self.sampling_rate
        self.dt_tol = self.dt_target * 1.5

        # 设置中文字体
        self.setup_chinese_font()

        # 初始化 UI（包含将所有动态 Artist 标记为 animated=True）
        self.init_ui()

        # FFT 结果
        self.fft_result = None
        self.last_fft_render = 0
        self.FFT_RENDER_INTERVAL = 1.0  # 每 1 秒更新一次 FFT

        # 启动动画（blit=True）
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=100,          # 100ms 刷新一次（约 10fps）
            blit=True,
            cache_frame_data=False
        )

        # 显示窗口（非阻塞）
        plt.show(block=False)

        # 尝试设置自定义图标
        try:
            icon_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "ecg_icon1.jpg")
            if os.path.exists(icon_path):
                win = self.fig.canvas.manager.window
                self.window_icon = tk.PhotoImage(file=icon_path)
                win.iconphoto(True, self.window_icon)
        except Exception as e:
            print(f"设置可视化窗口图标失败: {e}")

    def set_fft_result(self, fft_result):
        """外部调用，用于传入最新的 FFT 结果字典"""
        try:
            self.fft_result = fft_result
        except Exception as e:
            print(f"ECGVisualizer: 处理FFT结果错误: {str(e)}")

    def setup_chinese_font(self):
        """配置 matplotlib 支持中文"""
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False

    def init_ui(self):
        """创建并布局所有子图及 Artist，并将需要动态刷新的 Artist 打上 animated=True"""

        # 专业配色
        self.WAVEFORM_COLOR = '#0055A4'
        self.P_WAVE_COLOR = '#008000'
        self.T_WAVE_COLOR = '#C00000'
        self.QRS_COLOR = '#FFD700'
        self.BG_COLOR = '#F0F8FF'
        self.HR_COLOR = '#8B0000'
        self.ACCENT_COLOR = '#00BCD4'

        # 创建 Figure/Gridspec
        self.fig = plt.figure(figsize=(14, 9.5), facecolor=self.BG_COLOR)
        gs = plt.GridSpec(5, 1, height_ratios=[0.75, 0.02, 0.20, 0.03, 0.10])

        # ECG 波形、空白、FFT、空白、状态 共五块
        self.ax_ecg = plt.subplot(gs[0])
        self.ax_space1 = plt.subplot(gs[1])
        self.ax_fft = plt.subplot(gs[2])
        self.ax_space2 = plt.subplot(gs[3])
        self.ax_status = plt.subplot(gs[4])

        # 隐藏空白区
        self.ax_space1.set_axis_off()
        self.ax_space2.set_axis_off()

        # 窗口标题（此时空，稍后由 update_plot 动态更新）
        self.fig.canvas.manager.set_window_title('ECG实时分析系统')

        plt.subplots_adjust(hspace=0.35)

        # ----- ECG 波形区域 设置 -----
        self.ax_ecg.set_facecolor('white')
        # X 轴固定显示 [0, DISPLAY_WINDOW_SECONDS]，使用相对坐标
        self.ax_ecg.set_xlim(0, self.DISPLAY_WINDOW_SECONDS)
        self.ax_ecg.set_ylim(-1.0, 1.5)

        # 设置固定的刻度位置（相对 0–5 秒）
        major_locator = MultipleLocator(1.0)
        minor_locator = MultipleLocator(0.2)
        self.ax_ecg.xaxis.set_major_locator(major_locator)
        self.ax_ecg.xaxis.set_minor_locator(minor_locator)
        self.ax_ecg.yaxis.set_major_locator(MultipleLocator(0.5))
        self.ax_ecg.yaxis.set_minor_locator(MultipleLocator(0.1))
        self.ax_ecg.grid(which='minor', color='#D3D3D3', linewidth=0.5, alpha=0.5)
        self.ax_ecg.grid(which='major', color='#D3D3D3', linewidth=1.0, alpha=0.7)

        self.ax_ecg.set_xlabel('时间 (秒) [相对窗口]', fontsize=10)
        self.ax_ecg.set_ylabel('电压 (mV)', fontsize=10)

        # 心率文字（左上角）
        self.hr_text = self.ax_ecg.text(
            0.02, 0.98, '',
            transform=self.ax_ecg.transAxes,
            fontsize=24, fontweight='bold',
            color=self.HR_COLOR, ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgrey', boxstyle='round,pad=0.5')
        )
        self.hr_text.set_animated(True)

        # ----- FFT 区域 设置 -----
        self.ax_fft.set_facecolor('white')
        self.ax_fft.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        self.ax_fft.set_xlabel('频率 (Hz)', fontsize=9)
        self.ax_fft.set_ylabel('功率', fontsize=9)
        self.ax_fft.set_title('频谱分析 (自动更新)', fontsize=10, pad=10)
        self.ax_fft.set_xlim(0, 50)
        self.ax_fft.set_ylim(0, None)

        # ----- 状态栏 底部 -----
        self.ax_status.set_facecolor('#E8E8E8')
        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])

        # 左侧状态文字
        self.status_text = self.ax_status.text(
            0.01, 0.5, '状态: 等待数据.',
            fontsize=10, va='center', ha='left'
        )
        self.status_text.set_animated(True)

        # 中央时间文字（实时绝对时间）
        self.time_text = self.ax_status.text(
            0.5, 0.5, '时间: 0.00秒',
            fontsize=10, va='center', ha='center'
        )
        self.time_text.set_animated(True)

        # ----- 预创建并标记所有动态 Artist -----

        # 1. ECG 主波形曲线
        self.line, = self.ax_ecg.plot(
            [], [], linewidth=1.5, color=self.WAVEFORM_COLOR, animated=True
        )

        # 2. R 波散点
        self.r_peak_plot, = self.ax_ecg.plot(
            [], [], 'o', markersize=8, color='red', alpha=0.8, animated=True
        )
        # 3. P 波散点
        self.p_wave_plot, = self.ax_ecg.plot(
            [], [], 'o', markersize=6, color=self.P_WAVE_COLOR, alpha=0.8, animated=True
        )
        # 4. T 波散点
        self.t_wave_plot, = self.ax_ecg.plot(
            [], [], 'o', markersize=6, color=self.T_WAVE_COLOR, alpha=0.8, animated=True
        )

        # 5. QRS 区间 Patch 列表（可动态清除/重绘）
        self.qrs_patches = []

        # 6. T 波与 R 波虚线连接列表
        self.t_connection_lines = []

        # ----- FFT 相关的动态元素 -----
        # FFT 主线
        self.fft_line, = self.ax_fft.plot(
            [], [], color='#1E88E5', linewidth=1.5, animated=True
        )
        # FFT 峰值点
        self.peak_line, = self.ax_fft.plot(
            [], [], 'ro', markersize=5, animated=True
        )
        # FFT 心率注释
        self.hr_annotation = self.ax_fft.text(
            0, 0, '',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8),
            animated=True
        )

    def add_data(self, new_time, new_voltage, r_peaks, components):
        """
        由 DataProcessor 调用，传入新的时刻电压与检测到的 R/P/T/QRS 信息，
        并对时间空隙做插值补齐后存入 time_data/voltage_data。
        """
        with self.data_lock:
            if hasattr(self, 'paused') and self.paused:
                return
            if not isinstance(new_time, (int, float)) or not isinstance(new_voltage, (int, float)):
                return

            if not self.data_started:
                self.data_started = True
                self.status_text.set_text('状态: 数据接收中.')

            # 如果已有上一个点，检查是否需要插值补齐
            if self.t_prev is not None:
                dt = new_time - self.t_prev
                if dt > self.dt_tol:
                    # 计算需插值的虚拟点数
                    N_virtual = int(round(dt / self.dt_target)) - 1
                    for k in range(1, N_virtual + 1):
                        t_virtual = self.t_prev + k * (dt / (N_virtual + 1))
                        v_virtual = self.v_prev + (new_voltage - self.v_prev) * (k / (N_virtual + 1))
                        # 插入虚拟点到绘图和分析缓冲
                        self.time_data.append(t_virtual)
                        self.voltage_data.append(v_virtual)

            # 插入当前真实采样点
            self.time_data.append(new_time)
            self.voltage_data.append(new_voltage)
            self.t_prev = new_time
            self.v_prev = new_voltage

            # 更新心率、R/P/T/QRS/Components
            self.current_time = new_time
            self.r_peaks = r_peaks.copy()
            self.components = {
                'P': components.get('P', []).copy(),
                'QRS': components.get('QRS', []).copy(),
                'T': components.get('T', []).copy()
            }

            self.time_text.set_text(f'时间: {self.current_time:.2f}秒')

    def update_plot(self, frame):
        """
        每帧只更新发生变化的 artist，通过 blit 机制进行局部重绘。
        同时根据 time_data/voltage_data 绘制滚动波形和组件标记。
        """
        artists_to_update = []

        with self.data_lock:
            if not self.time_data:
                # 无数据时，仅返回已有动画对象，避免报错
                baseline = [
                    self.line,
                    self.r_peak_plot, self.p_wave_plot, self.t_wave_plot,
                    self.fft_line, self.peak_line,
                    self.hr_text, self.hr_annotation,
                    self.status_text, self.time_text
                ]
                return baseline

            time_arr = np.array(self.time_data, dtype=float)
            volt_arr = np.array(self.voltage_data, dtype=float)

            # 当前绝对时间窗口端点
            end_t = self.current_time
            start_t = max(0, end_t - self.DISPLAY_WINDOW_SECONDS)

            # 更新标题栏：显示绝对时间范围
            title_text = f"ECG 实时分析 | 窗口: {start_t:.2f} 秒 ～ {end_t:.2f} 秒"
            self.ax_ecg.set_title(title_text, fontsize=10, pad=12)
            artists_to_update.append(self.ax_ecg.title)

            # 取可见区间的数据
            mask = (time_arr >= start_t) & (time_arr <= end_t)
            if not np.any(mask):
                # 窗口内无数据，则清空所有波形和标记
                self.line.set_data([], [])
                artists_to_update.append(self.line)

                self.r_peak_plot.set_data([], [])
                self.p_wave_plot.set_data([], [])
                self.t_wave_plot.set_data([], [])
                artists_to_update += [self.r_peak_plot, self.p_wave_plot, self.t_wave_plot]

                for patch in self.qrs_patches:
                    patch.remove()
                self.qrs_patches.clear()

                for ln in self.t_connection_lines:
                    ln.remove()
                self.t_connection_lines.clear()

                return artists_to_update + [
                    self.fft_line, self.peak_line,
                    self.hr_annotation, self.status_text, self.time_text
                ]

            vis_times_abs = time_arr[mask]
            vis_volts = volt_arr[mask]
            # 计算相对窗口内时刻 [0, DISPLAY_WINDOW_SECONDS]
            rel_times = vis_times_abs - start_t

            # —— 1. 更新 ECG 主曲线 ——
            self.line.set_data(rel_times, vis_volts)
            artists_to_update.append(self.line)

            # —— 2. 更新 R 波散点 ——
            rx_rel, ry = [], []
            for r_t in self.r_peaks:
                if start_t <= r_t <= end_t:
                    idx = np.argmin(np.abs(vis_times_abs - r_t))
                    rx_rel.append(r_t - start_t)
                    ry.append(vis_volts[idx])
            if rx_rel:
                self.r_peak_plot.set_data(rx_rel, ry)
            else:
                self.r_peak_plot.set_data([], [])
            artists_to_update.append(self.r_peak_plot)

            # —— 3. 更新 P 波散点 ——
            px_rel, py = [], []
            for p_t in self.components.get('P', []):
                if start_t <= p_t <= end_t:
                    idx = np.argmin(np.abs(vis_times_abs - p_t))
                    px_rel.append(p_t - start_t)
                    py.append(vis_volts[idx])
            if px_rel:
                self.p_wave_plot.set_data(px_rel, py)
            else:
                self.p_wave_plot.set_data([], [])
            artists_to_update.append(self.p_wave_plot)

            # —— 4. 更新 T 波散点 ——
            tx_rel, ty = [], []
            for t_t in self.components.get('T', []):
                if start_t <= t_t <= end_t:
                    idx = np.argmin(np.abs(vis_times_abs - t_t))
                    tx_rel.append(t_t - start_t)
                    ty.append(vis_volts[idx])
            if tx_rel:
                self.t_wave_plot.set_data(tx_rel, ty)
            else:
                self.t_wave_plot.set_data([], [])
            artists_to_update.append(self.t_wave_plot)

            # —— 5. 更新 QRS 区间 Patch ——
            for patch in self.qrs_patches:
                patch.remove()
            self.qrs_patches.clear()
            for qrs_start, qrs_end in self.components.get('QRS', []):
                if qrs_end >= start_t and qrs_start <= end_t:
                    vs = max(qrs_start, start_t)
                    ve = min(qrs_end, end_t)
                    if vs < ve:
                        patch = self.ax_ecg.axvspan(
                            vs - start_t, ve - start_t,
                            color=self.QRS_COLOR, alpha=0.3,
                            animated=True
                        )
                        self.qrs_patches.append(patch)
                        artists_to_update.append(patch)

            # —— 6. 更新 T→R 虚线连接 ——
            for ln in self.t_connection_lines:
                ln.remove()
            self.t_connection_lines.clear()
            for r_t in self.r_peaks:
                if start_t <= r_t <= end_t:
                    closest_t = None
                    mind = float('inf')
                    for t_t in self.components.get('T', []):
                        d = abs(t_t - r_t)
                        if t_t > r_t and d < mind and d < 0.5:
                            mind = d
                            closest_t = t_t
                    if closest_t is not None:
                        r_rel = r_t - start_t
                        t_rel = closest_t - start_t
                        r_idx = np.argmin(np.abs(vis_times_abs - r_t))
                        t_idx = np.argmin(np.abs(vis_times_abs - closest_t))
                        y_r = vis_volts[r_idx]
                        y_t = vis_volts[t_idx]
                        ln, = self.ax_ecg.plot(
                            [r_rel, t_rel],
                            [y_r, y_t],
                            '--', color='gray', alpha=0.4, linewidth=1,
                            animated=True
                        )
                        self.t_connection_lines.append(ln)
                        artists_to_update.append(ln)

            # —— 7. 更新心率文字 ——
            if self.r_peaks and len(self.r_peaks) >= 2:
                try:
                    rr = self.r_peaks[-1] - self.r_peaks[-2]
                    hr = 60.0 / rr
                    self.hr_text.set_text(f"{hr:.0f} BPM")
                except:
                    self.hr_text.set_text("--")
            else:
                self.hr_text.set_text("--")
            artists_to_update.append(self.hr_text)

            # —— 8. 更新 FFT 频谱 ——
            now = time.time()
            if (now - self.last_fft_render >= self.FFT_RENDER_INTERVAL
                    and self.fft_result):
                try:
                    freqs = self.fft_result['freqs']
                    psd = self.fft_result['psd']
                    hr = self.fft_result['hr']

                    # FFT 主线
                    self.fft_line.set_data(freqs, psd)
                    artists_to_update.append(self.fft_line)

                    # 如果有心率信息
                    if hr is not None:
                        hr_freq = hr / 60.0
                        band = (freqs >= 0.8) & (freqs <= 3.0)
                        f_hr = freqs[band]
                        p_hr = psd[band]
                        if len(f_hr) > 0:
                            peak_idx = np.argmin(np.abs(f_hr - hr_freq))
                            px = f_hr[peak_idx]
                            py = p_hr[peak_idx]
                            self.peak_line.set_data([px], [py])
                            self.hr_annotation.set_text(f'{hr:.0f} BPM')
                            self.hr_annotation.set_position((px, py))
                            self.hr_annotation.set_visible(True)
                            artists_to_update += [self.peak_line, self.hr_annotation]
                            self.ax_fft.set_title(
                                f'频谱分析 (心率估算: {hr:.0f} BPM)',
                                fontsize=10, pad=10
                            )
                        else:
                            self.peak_line.set_data([], [])
                            self.hr_annotation.set_visible(False)
                            artists_to_update += [self.peak_line, self.hr_annotation]
                            self.ax_fft.set_title(
                                '频谱分析 (数据不足)',
                                fontsize=10, pad=10
                            )
                        # 自动拉伸 Y 轴
                        self.ax_fft.set_ylim(0, np.max(psd) * 1.1)
                    self.last_fft_render = now
                except Exception as e:
                    print(f"可视化FFT渲染错误: {str(e)}")

        # 最后，把状态栏文本加入更新列表
        artists_to_update += [self.status_text, self.time_text]

        return artists_to_update

    def close(self):
        """安全关闭可视化窗口"""
        self.running = False
        try:
            self.ax_ecg.cla()
            self.ax_fft.cla()
            self.ax_status.cla()
            plt.close(self.fig)
        except Exception as e:
            print(f"关闭可视化窗口时出现错误: {str(e)}")