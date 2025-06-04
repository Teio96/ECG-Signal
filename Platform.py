import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import csv
from datetime import datetime

from OnCSV import CSVDataCollector
from OnSPI import SPIDataCollector          # ← 新增：SPI 实时采集模块
from DataProcessor import DataProcessor
from ECGVisualizer import ECGVisualizer
from DataAnalyze import start_analysis, stop_analysis, analyze_data, plot_trends, get_recent_alerts, ECG_Analyzer
import DataAnalyze
import queue


class HeartTracerPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("HeartTracer 心电图分析平台")
        self.root.geometry("1200x700")
        self.root.minsize(400, 600)

        # 初始化ECG分析器
        self.analyzer = None

        # 设置主题变量
        self.dark_mode = False
        self.colors = self.get_theme_colors()

        # 初始化变量
        self.selected_file = ""
        self.collected_data = []
        self.analysis_running = False
        self.start_time = 0
        self.data_queue = queue.Queue()   # 用于传递原始数据
        self.result_queue = queue.Queue() # 用于传递处理结果
        self.visualizer = None            # 可视化窗口引用

        # **新增：是否使用 SPI 实时采集的标志**
        self.use_spi = tk.BooleanVar(value=False)

        # 默认保存路径：当前脚本目录下的 output/raw
        default_dir = os.path.join(os.path.dirname(__file__), "output", "raw")
        self.save_path = os.path.normpath(default_dir)
        os.makedirs(self.save_path, exist_ok=True)
        self.save_enabled = True

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 在初始化时，用 after 启动定时更新，但要保存 ID 以便后续取消
        self.after_elapsed_time_id = self.root.after(100, self.update_elapsed_time)
        self.after_data_display_id = self.root.after(100, self.update_data_display)

        # 配置字体
        self.default_font = ("Microsoft YaHei UI", 10)
        self.title_font = ("Microsoft YaHei UI", 16, "bold")

        # 创建主框架
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建标题和主题切换按钮
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 15))

        self.title_label = ttk.Label(
            self.header_frame,
            text="HeartTracer 心电图分析平台",
            font=self.title_font,
            foreground=self.colors["text"]
        )
        self.title_label.pack(side=tk.LEFT)

        self.theme_button = ttk.Button(
            self.header_frame,
            text="切换主题",
            command=self.toggle_theme,
            style='Accent.TButton'
        )
        self.theme_button.pack(side=tk.RIGHT, padx=5)

        self.help_button = ttk.Button(
            self.header_frame,
            text="帮助",
            command=self.show_help,
            style='Accent.TButton'
        )
        self.help_button.pack(side=tk.RIGHT, padx=5)

        # 文件选择部分
        self.file_frame = ttk.LabelFrame(self.main_frame, text="数据文件选择", padding=10)
        self.file_frame.pack(fill=tk.X, pady=(0, 15))

        # **新增：实时采集（SPI）复选框**
        cb_frame = ttk.Frame(self.file_frame)
        cb_frame.pack(fill=tk.X, pady=(0, 5))
        self.cb_use_spi = tk.Checkbutton(
            cb_frame,
            text="实时采集（SPI）",
            variable=self.use_spi,
            bg=self.colors["background"],
            fg=self.colors["text"],
            font=self.default_font,
            command=self.toggle_source
        )
        self.cb_use_spi.pack(side=tk.LEFT)

        # 文件路径显示和按钮
        self.path_label = ttk.Label(self.file_frame, text="未选择文件", foreground=self.colors["secondary_text"])
        self.path_label.pack(fill=tk.X, pady=(0, 5))

        button_frame = ttk.Frame(self.file_frame)
        button_frame.pack(fill=tk.X)

        self.btn_select = ttk.Button(
            button_frame,
            text="选择 CSV 文件",
            command=self.select_file,
            style='Accent.TButton'
        )
        self.btn_select.pack(side=tk.LEFT, padx=(0, 5))

        self.help_file_button = ttk.Button(
            button_frame,
            text="文件格式帮助",
            command=self.show_file_help,
            style='Info.TButton'
        )
        self.help_file_button.pack(side=tk.LEFT)

        # 分析控制部分
        self.control_frame = ttk.LabelFrame(self.main_frame, text="分析控制", padding=10)
        self.control_frame.pack(fill=tk.X, pady=(0, 15))

        control_btn_frame = ttk.Frame(self.control_frame)
        control_btn_frame.pack(fill=tk.X)

        self.btn_run = ttk.Button(
            control_btn_frame,
            text="启动分析",
            command=self.run_analysis,
            state=tk.DISABLED,
            style='Success.TButton'
        )
        self.btn_run.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_pause = ttk.Button(
            control_btn_frame,
            text="暂停",
            command=self.toggle_pause,
            style='Accent.TButton',
            state=tk.DISABLED
        )
        self.btn_pause.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_stop = ttk.Button(
            control_btn_frame,
            text="停止分析",
            command=self.stop_analysis,
            state=tk.DISABLED,
            style='Danger.TButton'
        )
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_save_setting = ttk.Button(
            control_btn_frame,
            text="存储设置",
            command=self.open_save_settings_window,
            state=tk.NORMAL,
            style='Accent.TButton'
        )
        self.btn_save_setting.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_export = ttk.Button(
            control_btn_frame,
            text="导出结果",
            command=self.export_data,
            state=tk.DISABLED,
            style='Accent.TButton'
        )
        self.btn_export.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_trends = ttk.Button(
            control_btn_frame,
            text="趋势图",
            command=self.show_trends,
            style='Accent.TButton',
            state=tk.DISABLED
        )
        self.btn_trends.pack(side=tk.LEFT)

        # 状态显示
        self.status_frame = ttk.Frame(self.control_frame)
        self.status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = ttk.Label(
            self.status_frame,
            text="就绪",
            foreground=self.colors["accent"],
            font=self.default_font
        )
        self.status_label.pack(side=tk.LEFT)

        self.elapsed_label = ttk.Label(
            self.status_frame,
            text="",
            foreground=self.colors["secondary_text"],
            font=self.default_font
        )
        self.elapsed_label.pack(side=tk.RIGHT)

        # 信息输出栏
        self.info_frame = ttk.LabelFrame(self.main_frame, text="信息输出栏", padding=10)
        self.info_frame.pack(fill=tk.X, pady=(0, 15))

        self.info_text = tk.Text(
            self.info_frame,
            height=8,
            bg=self.colors["realtime_bg"],
            fg=self.colors["text"],
            font=("Consolas", 10),
            borderwidth=0,
            insertbackground=self.colors["text"]
        )
        info_scrollbar = ttk.Scrollbar(self.info_frame, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        self.info_text.tag_configure("alert", foreground="red", font=("Consolas", 10, "bold"))
        self.info_text.insert(tk.END, "等待操作...\n")
        self.info_text.configure(state=tk.DISABLED)

        # 结果展示部分
        self.result_frame = ttk.LabelFrame(self.main_frame, text="分析结果", padding=10)
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("ID", "P波位置", "P波幅度", "QRS区间", "T波位置", "T波幅度", "心率(BPM)", "基线电压")
        self.result_tree = ttk.Treeview(
            self.result_frame,
            columns=columns,
            show="headings",
            selectmode="extended",
            style="Treeview"
        )
        for col in columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=120, anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(self.result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_tree.pack(fill=tk.BOTH, expand=True)

        self.stats_frame = ttk.Frame(self.result_frame)
        self.stats_frame.pack(fill=tk.X, pady=(10, 0))
        self.stats_label = ttk.Label(
            self.stats_frame,
            text="0 条记录",
            foreground=self.colors["text"],
            font=self.default_font
        )
        self.stats_label.pack(side=tk.RIGHT)

        # 底部状态栏
        self.footer_frame = ttk.Frame(self.main_frame, relief=tk.SUNKEN, padding=(5, 2))
        self.footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.footer_label = ttk.Label(
            self.footer_frame,
            text="就绪",
            foreground=self.colors["secondary_text"]
        )
        self.footer_label.pack(side=tk.LEFT)

        self.version_label = ttk.Label(
            self.footer_frame,
            text="HeartTracer v2.0",
            foreground=self.colors["secondary_text"]
        )
        self.version_label.pack(side=tk.RIGHT)

        # 应用主题
        self.apply_theme()

        # 初始化定时器
        self.root.after(100, self.update_elapsed_time)
        self.root.after(100, self.update_data_display)

        self.analysis_paused = False


    def get_theme_colors(self):
        """获取当前主题的颜色方案（不变）"""
        if self.dark_mode:
            return {
                "background": "#121212",
                "foreground": "#2c3e50",
                "text": "#e0e0e0",
                "secondary_text": "#757575",
                "accent": "#00bcd4",
                "success": "#4caf50",
                "danger": "#f44336",
                "tree_bg": "#1e1e1e",
                "tree_fg": "#ffffff",
                "tree_sel": "#37474f",
                "header_bg": "#2d2d2d",
                "button_bg": "#616161",
                "button_fg": "#000000",
                "realtime_bg": "#0d0d0d",
                "scrollbar": "#424242",
                "scrollbar_grip": "#000000",
                "tree_header_bg": "#424242",
                "tree_header_fg": "#000000",
                "tree_border": "#616161",
                "warning": "#ff9800"
            }
        else:
            return {
                "background": "#f5f5f7",
                "foreground": "#ffffff",
                "text": "#000000",
                "secondary_text": "#616161",
                "accent": "#2196f3",
                "success": "#4caf50",
                "danger": "#f44336",
                "tree_bg": "#ffffff",
                "tree_fg": "#000000",
                "tree_sel": "#bbdefb",
                "header_bg": "#f5f5f5",
                "button_bg": "#e0e0e0",
                "button_fg": "#333333",
                "realtime_bg": "#f8f8f8",
                "scrollbar": "#e0e0e0",
                "scrollbar_grip": "#bdbdbd",
                "tree_header_bg": "#e0e0e0",
                "tree_header_fg": "#212121",
                "tree_border": "#616161",
                "warning": "#ff9800",
            }

    def apply_theme(self):
        """应用当前主题到所有控件（不变）"""
        colors = self.colors
        style = ttk.Style()

        style.configure('.',
                        background=colors["background"],
                        foreground=colors["text"],
                        font=self.default_font)

        button_styles = [
            ('TButton', 'default'),
            ('Accent.TButton', 'accent'),
            ('Success.TButton', 'success'),
            ('Danger.TButton', 'danger'),
            ('Info.TButton', '#9b59b6')
        ]

        for style_name, color_key in button_styles:
            if color_key in colors:
                bg_color = colors[color_key] if color_key != 'default' else colors["button_bg"]
            else:
                bg_color = color_key

            style.configure(style_name,
                            padding=8,
                            foreground=colors["button_fg"],
                            background=bg_color,
                            bordercolor=colors["background"])

            style.map(style_name,
                      foreground=[('active', colors["accent"]),
                                  ('disabled', colors["secondary_text"])],
                      background=[('active', bg_color),
                                  ('disabled', colors["background"])])

        style.configure('Treeview',
                        font=('Consolas', 10),
                        rowheight=28,
                        background=colors["tree_bg"],
                        foreground=colors["tree_fg"],
                        fieldbackground=colors["tree_bg"])

        style.configure('Treeview.Heading',
                        font=('Microsoft YaHei UI', 10, 'bold'),
                        background=colors["tree_header_bg"],
                        foreground=colors["tree_header_fg"],
                        relief="flat",
                        borderwidth=2,
                        bordercolor=colors["tree_border"])

        style.map('Treeview',
                  background=[('selected', colors["tree_sel"])])

        style.map('Treeview.Heading',
                  background=[('active', colors["tree_header_bg"])])

        style.configure("Vertical.TScrollbar",
                        background=colors["scrollbar"],
                        troughcolor=colors["background"],
                        gripcount=0,
                        arrowsize=14)
        style.map("Vertical.TScrollbar",
                  gripcolor=[('', colors["scrollbar_grip"])])

        style.configure('TLabelframe',
                        background=colors["background"],
                        bordercolor=colors["accent"])
        style.configure('TLabelframe.Label',
                        background=colors["background"],
                        foreground=colors["text"])

        self.root.configure(background=colors["background"])
        self.main_frame.configure(style='TFrame')
        if hasattr(self, 'info_text'):
            self.info_text.configure(
                bg=colors["realtime_bg"],
                fg=colors["text"],
                insertbackground=colors["text"]
            )

        self.title_label.configure(foreground=colors["text"])

    def toggle_theme(self):
        """切换深色/浅色主题（不变）"""
        self.dark_mode = not self.dark_mode
        self.colors = self.get_theme_colors()
        self.apply_theme()

    def toggle_source(self):
        """
        切换数据源模式：
        - 如果勾选了“实时采集（SPI）”，禁用 CSV 文件选择，并直接启用“启动分析”按钮；
        - 如果取消勾选，恢复需要手动选择 CSV 文件，置为“未选择文件”状态。
        """
        if self.use_spi.get():
            # SPI 模式
            self.btn_select.config(state=tk.DISABLED)
            self.btn_run.config(state=tk.NORMAL)
            self.path_label.config(text="SPI 模式已选择", foreground=self.colors["text"])
            self.status_label.config(text="SPI 实时采集模式", foreground=self.colors["accent"])
        else:
            # CSV 模式
            self.btn_select.config(state=tk.NORMAL)
            self.btn_run.config(state=tk.DISABLED)
            self.selected_file = ""
            self.path_label.config(text="未选择文件", foreground=self.colors["secondary_text"])
            self.status_label.config(text="请先选择 CSV 文件", foreground=self.colors["danger"])


    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="选择 ECG 数据文件",
            filetypes=[("CSV 文件", "*.csv")],
            initialdir=os.getcwd()
        )
        if file_path:
            self.selected_file = file_path
            self.path_label.config(text=os.path.basename(file_path), foreground=self.colors["text"])
            if not self.use_spi.get():
                self.btn_run.config(state=tk.NORMAL)
                self.status_label.config(text="文件已选择，准备分析", foreground=self.colors["success"])
                self.footer_label.config(text=f"已选择文件: {os.path.basename(file_path)}")
        else:
            self.selected_file = ""
            self.path_label.config(text="未选择文件", foreground=self.colors["secondary_text"])
            if not self.use_spi.get():
                self.btn_run.config(state=tk.DISABLED)
                self.status_label.config(text="请选择CSV文件", foreground=self.colors["danger"])


    def run_analysis(self):
        """
        启动分析流程：
        - 如果是 SPI 模式，直接实例化 SPIDataCollector；
        - 如果是 CSV 模式，检查文件是否有效，然后实例化 CSVDataCollector。
        """
        # **如果是 CSV 模式，先验证文件有效性**
        if not self.use_spi.get():
            if not os.path.isfile(self.selected_file):
                messagebox.showerror("错误", "所选文件无效！")
                return

        # 清除之前的结果
        self.collected_data = []
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self.stats_label.config(text="0 条记录")

        # 启动分析，并在“信息输出栏”打印提示
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "分析开始...\n")
        self.info_text.configure(state=tk.DISABLED)

        # 数据记录相关变量
        self.data_file = None
        self.data_file_path = ""
        self.last_write_time = 0
        self.write_buffer = []

        # 更新UI状态
        self.analysis_running = True
        self.start_time = time.time()
        self.status_label.config(text="分析进行中...", foreground=self.colors["accent"])
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_export.config(state=tk.DISABLED)
        self.btn_trends.config(state=tk.NORMAL)
        self.btn_save_setting.config(state=tk.DISABLED)
        self.footer_label.config(text="分析进行中...")

        # 如果已有旧的分析器则先停止
        if hasattr(self, 'analyzer') and self.analyzer is not None:
            try:
                self.analyzer.stop()
                self.analyzer.analysis_thread.join(timeout=1.0)
            except Exception:
                pass
        self.analyzer = None

        # 如果存在旧的趋势窗口，先销毁
        if hasattr(self, 'trend_window') and getattr(self.trend_window, "winfo_exists", lambda: False)():
            try:
                self.trend_window.destroy()
            except Exception:
                pass

        # 清空队列
        while not self.data_queue.empty():
            self.data_queue.get()
        while not self.result_queue.empty():
            self.result_queue.get()

        try:
            # —— 根据模式选择 DataCollector —— #
            if self.use_spi.get():
                # SPI 实时采集模式
                self.data_collector = SPIDataCollector()
            else:
                # CSV 文件模式
                self.data_collector = CSVDataCollector(self.selected_file)

            # 实例化 DataProcessor
            self.data_processor = DataProcessor(
                self.data_queue,
                self.result_queue,
                self.realtime_data_callback,
                self.handle_raw_data,
                self.data_collector.sampling_rate
            )

            # 创建 ECG_Analyzer，用于报警
            self.analyzer = ECG_Analyzer(alert_callback=self.handle_alert)
            self.analyzer.sampling_rate = self.data_collector.sampling_rate
            DataAnalyze.global_analyzer = self.analyzer
            self.analyzer.start()

            # 创建可视化界面
            self.visualizer = ECGVisualizer(self.data_collector.sampling_rate)

            # 设置 FFT 回调
            self.data_processor.fft_callback = self.visualizer.set_fft_result

            # 将可视化器连接到处理器
            self.data_processor.set_visualization_callback(self.visualizer.add_data)

            # 创建数据文件并写入表头和采样率（仅当用户开启“保存功能”时才会写入原始数据）
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ecg_data_{timestamp}.csv"
            full_path = os.path.join(self.save_path, filename)
            self.data_file_path = full_path
            self.data_file = open(self.data_file_path, 'w', newline='')
            self.data_file.write(os.path.basename(self.data_file_path) + "\n")
            self.data_file.write(f"{self.data_collector.sampling_rate}\n")
            self.last_flush_time = time.time()
            self.stream_finished = False
            self.add_realtime_message(f"数据记录已启动，文件: {self.data_file_path}")

            # 启动采集线程
            self.collector_thread = threading.Thread(
                target=self.data_collector.start_stream,
                args=(self.data_queue,),
                daemon=True
            )
            # 启动处理线程
            self.processor_thread = threading.Thread(
                target=self.data_processor.process_data,
                daemon=True
            )

            self.collector_thread.start()
            self.processor_thread.start()

            # 如果是 CSV 模式，禁止再选择新文件；SPI 模式下本来就已禁用
            if not self.use_spi.get():
                self.btn_select.config(state=tk.DISABLED)

            # 启动展示循环
            self.update_data_display()

        except Exception as e:
            self.status_label.config(text=f"启动分析失败: {str(e)}")
            self.add_realtime_message(f"错误: {str(e)}")
            self.analysis_completed()


    # 新增：Platform 专门写“原始（每个采样点）电压”到 CSV
    def handle_raw_data(self, timestamp, voltage):
        if not self.save_enabled:
            return
        try:
            if not hasattr(self, 'data_file') or self.data_file is None:
                os.makedirs(self.save_path, exist_ok=True)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ecg_data_{timestamp_str}.csv"
                full_path = os.path.join(self.save_path, filename)
                self.data_file_path = full_path
                self.data_file = open(full_path, 'w', newline='')
                self.data_file.write(os.path.basename(full_path) + "\n")
                self.data_file.write(f"{self.data_collector.sampling_rate}\n")
                self.last_flush_time = time.time()
                self.add_realtime_message(f"数据记录已启动，文件: {self.data_file_path}")

            self.data_file.write(f"{voltage}\n")
            current_time = time.time()
            if current_time - self.last_flush_time > 1.0:
                self.data_file.flush()
                self.last_flush_time = current_time
        except Exception as e:
            print(f"[Platform] 原始数据写 CSV 失败: {e}")


    def update_data_display(self):
        """更新数据展示（修复processed变量未定义问题）"""
        processed = False
        if self.analysis_running and not self.analysis_paused:
            while not self.result_queue.empty():
                try:
                    data_row = self.result_queue.get_nowait()
                    self.add_to_results_table(data_row)
                    processed = True
                except queue.Empty:
                    break

        if (self.analysis_running or not processed) and self.root.winfo_exists():
            try:
                if hasattr(self, 'after_data_display_id'):
                    self.root.after_cancel(self.after_data_display_id)
            except Exception:
                pass
            self.after_data_display_id = self.root.after(100, self.update_data_display)


    def realtime_data_callback(self, data_row):
        """接收实时数据的回调函数（在处理器线程中调用）"""
        print(f"[PLATFORM] 接收到数据行: {data_row}")
        self.result_queue.put(data_row)


    def add_alert_message(self, message):
        """添加报警消息到实时数据框"""
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.insert(tk.END, f"!ALERT! {message}\n", "alert")
        self.info_text.see(tk.END)
        self.info_text.configure(state=tk.DISABLED)


    def add_realtime_message(self, message):
        """添加消息到实时数据框"""
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.insert(tk.END, f"\n{message}\n")
        self.info_text.see(tk.END)
        self.info_text.configure(state=tk.DISABLED)


    def add_to_results_table(self, data_row):
        """添加单行数据到结果表格"""
        self.collected_data.append(data_row)
        if len(data_row) < 8:
            data_row = data_row + ['-'] * (8 - len(data_row))
        self.result_tree.insert("", "end", values=data_row)
        count = len(self.result_tree.get_children())
        self.stats_label.config(text=f"{count} 条记录")
        self.result_tree.yview_moveto(1.0)


    def analysis_completed(self):
        """分析完成后更新UI状态"""
        self.analysis_running = False
        self.btn_run.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.NORMAL if self.collected_data else tk.DISABLED)
        self.btn_select.config(state=tk.NORMAL if not self.use_spi.get() else tk.DISABLED)


    def stop_analysis(self):
        """停止分析过程（增强版）"""
        self.analysis_running = False
        self.analysis_paused = False
        self.btn_pause.config(text="暂停", state=tk.DISABLED)
        if hasattr(self, 'data_collector'):
            self.data_collector.stop_stream()

        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        if hasattr(self, 'visualizer') and self.visualizer:
            try:
                self.visualizer.close()
                print("可视化界面已关闭")
            except Exception as e:
                print(f"关闭可视化界面错误: {str(e)}")

        if self.data_file:
            try:
                self.data_file.flush()
                self.data_file.close()
                self.add_realtime_message(f"数据已保存到: {self.data_file_path}")
            except Exception as e:
                print(f"关闭数据文件失败: {str(e)}")
            finally:
                self.data_file = None

        if hasattr(self, 'collector_thread') and self.collector_thread.is_alive():
            self.collector_thread.join(timeout=1.0)
        if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=1.0)

        if hasattr(self, 'analyzer') and self.analyzer is not None:
            self.analyzer.stop()
            try:
                self.analyzer.analysis_thread.join(timeout=1.0)
            except Exception:
                pass

        self.status_label.config(text="分析已停止", foreground=self.colors["danger"])
        self.footer_label.config(text="分析已停止")
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_run.config(state=tk.NORMAL)
        self.btn_save_setting.config(state=tk.NORMAL)
        self.add_realtime_message("分析已停止！")
        self.analysis_completed()


    def toggle_pause(self):
        """切换暂停状态"""
        self.analysis_paused = not self.analysis_paused

        if self.analysis_paused:
            self.status_label.config(text="分析已暂停", foreground=self.colors["warning"])
            self.btn_pause.config(text="继续")
            self.footer_label.config(text="分析已暂停")
            if hasattr(self, 'data_processor'):
                self.data_processor.pause()
        else:
            self.status_label.config(text="分析恢复中...", foreground=self.colors["accent"])
            self.btn_pause.config(text="暂停")
            self.footer_label.config(text="分析恢复中...")
            if hasattr(self, 'data_processor'):
                self.data_processor.resume()

        self.btn_run.config(state=tk.DISABLED if self.analysis_running else tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL if self.analysis_running else tk.DISABLED)


    def update_elapsed_time(self):
        """更新经过时间显示"""
        if self.analysis_running and self.root.winfo_exists():
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            self.elapsed_label.config(text=f"用时: {mins:02d}:{secs:02d}")

            try:
                if hasattr(self, 'after_elapsed_time_id'):
                    self.root.after_cancel(self.after_elapsed_time_id)
            except Exception:
                pass
            self.after_elapsed_time_id = self.root.after(1000, self.update_elapsed_time)


    def export_data(self):
        """导出分析结果到 CSV 文件（不变）"""
        if not self.collected_data:
            messagebox.showinfo("导出", "没有数据可导出")
            return

        default_export_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "output", "analysis_and_alert")
        )
        os.makedirs(default_export_dir, exist_ok=True)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv")],
            title="保存导出结果",
            initialdir=default_export_dir,
            initialfile=f"ecg_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_analysis.csv"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "P波位置", "P波幅度", "QRS区间", "T波位置", "T波幅度", "心率(BPM)", "基线电压"])
                writer.writerows(self.collected_data)
            messagebox.showinfo("导出成功", f"数据已成功导出到:\n{file_path}")
            self.footer_label.config(text=f"数据已导出到: {os.path.basename(file_path)}")

            if messagebox.askyesno("导出报警", "是否同时导出报警历史？"):
                alert_file = os.path.splitext(file_path)[0] + "_alerts.csv"
                try:
                    with open(alert_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["时间戳", "报警类型", "报警消息"])
                        for alert in get_recent_alerts(100):
                            writer.writerow([alert['timestamp'], alert['type'], alert['message']])
                    messagebox.showinfo("导出成功", f"报警历史已导出到:\n{alert_file}")
                except Exception as e:
                    messagebox.showerror("导出错误", f"导出报警历史时出错:\n{str(e)}")

        except Exception as e:
            messagebox.showerror("导出错误", f"导出数据时出错:\n{str(e)}")


    def show_file_help(self):
        """显示 CSV 文件格式帮助（不变）"""
        help_text = """支持的CSV文件格式要求：

1. 第一行：导联名称（如 "I", "II", "V1" 等）
2. 第二行：采样率（单位：Hz，如 "250"）
3. 从第三行开始：电压数据（单位：mV）
4. 只能识别CSV文件第一列数据

示例文件内容：
I
250
-0.125
0.015
0.135
-0.075
...
"""
        messagebox.showinfo("CSV文件格式说明", help_text)


    def show_help(self):
        """显示平台帮助信息（不变）"""
        help_text = """HeartTracer 心电图分析平台使用指南

1. 选择文件：点击"选择CSV文件"按钮导入心电图数据
2. 启动分析：文件导入后点击"启动分析"开始分析过程
3. 暂停/继续：分析过程开始后可以暂停或继续分析过程
4. 保存设置：分析过程中可选择保存原始数据于CSV文件
5. 心电识别：心电信号波形识别显示在"分析结果"区域
6. 实时波形：心电信号实时波形及其频谱在弹窗中显示
7. 报警系统：检测到异常时显示红色报警信息
8. 趋势分析：点击"趋势图"查看心电图参数变化趋势
9. 导出数据：点击"导出结果"将分析结果保存为CSV文件

平台功能：
- 实时心电图波形显示
- P波、QRS波群、T波自动检测
- 心率计算
- 心电信号频谱分析
- 基线电压分析
- 实时数据显示
- 异常报警系统（心动过缓、心动过速等）
- 心电图参数趋势分析
- 数据导出功能
- 深色/浅色主题切换
"""
        messagebox.showinfo("平台使用帮助", help_text)


    def handle_alert(self, alert):
        """处理报警事件（不变）"""
        self.add_alert_message(f"报警! {alert['type']}: {alert['message']}")
        self.status_label.config(text=f"报警: {alert['type']}", foreground=self.colors["danger"])
        self.footer_label.config(text=f"报警: {alert['type']} - {alert['message']}")
        self.root.bell()


    def show_trends(self):
        """生成并显示 ECG 趋势图（不变）"""
        try:
            if hasattr(self, 'trend_window') and getattr(self.trend_window, "winfo_exists", lambda: False)():
                self.trend_window.destroy()

            trends_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "output", "trends")
            )
            os.makedirs(trends_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ecg_trends_{timestamp}.png"
            file_path = os.path.join(trends_dir, filename)

            plot_trends(file_path)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"趋势图文件 {file_path} 未生成")

            self.show_image_window(file_path)

        except Exception as e:
            messagebox.showerror("错误", f"生成趋势图失败: {str(e)}")


    def show_image_window(self, image_path):
        """在新窗口中显示图像，并提供关闭按钮（不变）"""
        if hasattr(self, 'trend_window') and self.trend_window.winfo_exists():
            self.trend_window.lift()
            return

        self.trend_window = tk.Toplevel(self.root)
        self.trend_window.title("ECG趋势分析")
        self.trend_window.geometry("800x800")
        self.trend_window.configure(background=self.colors["background"])
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "ecg_icon.png")
            if os.path.exists(icon_path):
                self.trend_window.iconphoto(True, tk.PhotoImage(file=icon_path))
        except Exception as e:
            print(f"设置趋势图窗口图标失败: {e}")

        try:
            img = tk.PhotoImage(file=image_path)
        except Exception as e:
            messagebox.showerror("错误", f"加载趋势图文件失败: {str(e)}")
            close_btn = ttk.Button(self.trend_window, text="关闭", command=self.trend_window.destroy)
            close_btn.pack(pady=10)
            return

        self.trend_img = img
        container = tk.Frame(self.trend_window, bg=self.colors["background"])
        container.pack(fill=tk.BOTH, expand=True)
        v_scroll = ttk.Scrollbar(container, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll = ttk.Scrollbar(container, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        canvas = tk.Canvas(
            container,
            bg=self.colors["background"],
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set
        )
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.config(command=canvas.yview)
        h_scroll.config(command=canvas.xview)
        image_id = canvas.create_image(0, 0, anchor="nw", image=self.trend_img)
        canvas.config(scrollregion=canvas.bbox(image_id))
        close_btn = ttk.Button(self.trend_window, text="关闭", command=self.trend_window.destroy)
        close_btn.pack(pady=10)
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)


    def open_save_settings_window(self):
        """弹出子窗口，用于设置原始数据 CSV 的保存路径与开关（不变）"""
        if hasattr(self, 'save_settings_window') and self.save_settings_window.winfo_exists():
            self.save_settings_window.lift()
            return

        self.save_settings_window = tk.Toplevel(self.root)
        self.save_settings_window.title("存储设置")
        self.save_settings_window.geometry("520x160")
        self.save_settings_window.resizable(False, False)

        bg_color = '#F0F8FF'
        fg_color = '#000000'
        btn_padx = 10
        btn_pady = 5

        self.save_settings_window.configure(background=bg_color)
        container = tk.Frame(self.save_settings_window, bg=bg_color)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        lbl_prefix = tk.Label(
            container,
            text="原始数据保存路径：",
            bg=bg_color,
            fg=fg_color,
            font=("微软雅黑", 10)
        )
        lbl_prefix.grid(row=0, column=0, sticky="w")

        self.save_path_label = tk.Label(
            container,
            text=os.path.normpath(self.save_path),
            bg=bg_color,
            fg=fg_color,
            font=("微软雅黑", 10),
            wraplength=350,
            justify=tk.LEFT,
            anchor="w"
        )
        self.save_path_label.grid(row=0, column=1, sticky="w", padx=(5, 0))

        select_path_btn = ttk.Button(
            container,
            text="选择路径",
            command=self.select_save_directory,
            style='Accent.TButton'
        )
        select_path_btn.grid(row=1, column=0, pady=(btn_pady, 10), padx=(20,10), sticky="e")

        reset_btn = ttk.Button(
            container,
            text="恢复默认",
            command=self.restore_default_path,
            style='Accent.TButton'
        )
        reset_btn.grid(row=1, column=1, pady=(btn_pady, 10), padx=(10,20), sticky="w")

        self.save_enabled_var = tk.BooleanVar(value=self.save_enabled)
        self.save_checkbox = tk.Checkbutton(
            container,
            text="启用保存功能",
            variable=self.save_enabled_var,
            bg=bg_color,
            fg=fg_color,
            selectcolor=bg_color,
            font=("微软雅黑", 10)
        )
        self.save_checkbox.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.save_settings_window.protocol("WM_DELETE_WINDOW", self.on_close_save_settings_window)


    def restore_default_path(self):
        """将保存路径重置为默认值（不变）"""
        default_path = os.path.join(os.path.dirname(__file__), "output", "raw")
        self.save_path = os.path.normpath(default_path)
        try:
            os.makedirs(self.save_path, exist_ok=True)
        except Exception as e:
            messagebox.showerror("错误", f"创建默认保存目录失败:\n{e}")
            return
        if hasattr(self, 'save_path_label'):
            self.save_path_label.config(text=os.path.normpath(self.save_path))


    def select_save_directory(self):
        new_dir = filedialog.askdirectory(
            title="选择原始数据 CSV 存储目录",
            initialdir=self.save_path
        )
        if new_dir:
            normalized = os.path.normpath(new_dir)
            self.save_path = normalized
            self.save_enabled = self.save_enabled_var.get()
            if self.save_enabled:
                try:
                    os.makedirs(self.save_path, exist_ok=True)
                except Exception as e:
                    messagebox.showerror("错误", f"创建保存目录失败:\n{e}")
                    return
            self.save_path_label.config(text=os.path.normpath(self.save_path))
            self.save_settings_window.destroy()
            delattr(self, 'save_settings_window')


    def confirm_save_settings(self):
        """确认用户的保存设置（不变）"""
        self.save_enabled = self.save_enabled_var.get()
        if self.save_enabled:
            try:
                os.makedirs(self.save_path, exist_ok=True)
            except Exception as e:
                messagebox.showerror("错误", f"创建保存目录失败:\n{e}")
                return
        self.save_settings_window.destroy()
        delattr(self, 'save_settings_window')


    def on_close_save_settings_window(self):
        """关闭保存设置窗口时，先读取复选框状态再销毁（不变）"""
        self.save_enabled = self.save_enabled_var.get()
        if self.save_enabled:
            try:
                os.makedirs(os.path.normpath(self.save_path), exist_ok=True)
            except Exception as e:
                messagebox.showerror("错误", f"创建保存目录失败:\n{e}")
                return
        self.save_settings_window.destroy()
        delattr(self, 'save_settings_window')


    def on_closing(self):
        """用户点击窗口“×”按钮时的清理逻辑（不变，只做顺序调整）"""
        self.analysis_running = False
        self.analysis_paused = True

        try:
            if hasattr(self, 'after_data_display_id'):
                self.root.after_cancel(self.after_data_display_id)
        except Exception:
            pass

        try:
            if hasattr(self, 'after_elapsed_time_id'):
                self.root.after_cancel(self.after_elapsed_time_id)
        except Exception:
            pass

        try:
            if hasattr(self, 'analyzer') and self.analyzer is not None:
                self.analyzer.stop()
                self.analyzer.analysis_thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            if hasattr(self, 'data_processor'):
                self.data_processor.stop_processing()
            if hasattr(self, 'data_collector'):
                self.data_collector.stop_stream()
        except Exception:
            pass

        self.root.destroy()


def launch_platform():
    root = tk.Tk()
    try:
        icon_path = os.path.join(os.path.dirname(__file__), "ecg_icon.png")
        if os.path.exists(icon_path):
            root.iconphoto(True, tk.PhotoImage(file=icon_path))
    except Exception as e:
        print(f"设置图标失败: {e}")

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = HeartTracerPlatform(root)
    root.mainloop()


if __name__ == "__main__":
    launch_platform()