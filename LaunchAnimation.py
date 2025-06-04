import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QPixmap, QFontMetrics
from PyQt5.QtCore import Qt, QTimer, QPointF, QObject, pyqtSignal, QThread


class LoadWorker(QObject):
    finished = pyqtSignal()

    def run(self):
        time.sleep(6)  # 模拟加载过程
        self.finished.emit()


class ECGWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HeartTracer Launch")
        self.setFixedSize(1000, 500)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: black;")
        self.move_to_center()

        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.black)

        self.label_items = []
        self.cn_label_items = []
        self.footer = QLabel("Tracer Co. Ltd", self)
        self.footer.setStyleSheet("color: gray")
        self.footer.setFont(QFont("Arial", 10))
        self.footer.move(800, 470)

        self.text_en = "HeartTracer"
        self.text_cn = "追心者"

        self.init_text_animation()

        self.points = []
        self.index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_wave)
        self.timer.start(16)

        self.fade_alpha = 255
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.fade_out)

        self.start_loading()

    def move_to_center(self):
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def start_loading(self):
        # 创建 QThread 对象
        self.worker_thread = QThread()
        self.worker = LoadWorker()

        # 将 worker 移动到新线程
        self.worker.moveToThread(self.worker_thread)

        # 连接信号和槽
        self.worker.finished.connect(self.loading_completed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # 启动线程并执行任务
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

        # 安全超时机制
        QTimer.singleShot(7000, self.force_exit_if_stuck)

    def loading_completed(self):
        self.fade_timer.start(50)  # 在主线程中启动淡出

    def force_exit_if_stuck(self):
        if not self.fade_timer.isActive():
            self.fade_timer.start(50)

    def fade_out(self):
        self.fade_alpha -= 15
        if self.fade_alpha <= 0:
            self.fade_timer.stop()
            QApplication.instance().quit()
        self.update()

    def init_text_animation(self):
        delay = 200
        font_en = QFont("Arial", 32, QFont.Bold)
        font_cn = QFont("SimHei", 30, QFont.Bold)
        fm_en = QFontMetrics(font_en)
        fm_cn = QFontMetrics(font_cn)

        x_cursor_en = 100
        for i, char in enumerate(self.text_en):
            width = fm_en.width(char)

            # 使用工厂函数解决lambda闭包问题
            def make_lambda(c, x, y, f):
                return lambda: self.add_char(c, x, y, f)

            QTimer.singleShot(i * delay, make_lambda(char, x_cursor_en, 100, font_en))
            x_cursor_en += width

        x_cursor_cn = 120
        for j, char in enumerate(self.text_cn):
            width = fm_cn.width(char)

            def make_lambda(c, x, y, f):
                return lambda: self.add_char(c, x, y, f)

            QTimer.singleShot(j * delay + len(self.text_en) * delay, make_lambda(char, x_cursor_cn, 200, font_cn))
            x_cursor_cn += width

    def add_char(self, char, x, y, font):
        label = QLabel(char, self)
        label.setFont(font)
        label.setStyleSheet("color: cyan")
        label.move(x, y)
        label.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(0, 0, self.pixmap)
        if self.fade_alpha < 255:
            fade_color = QColor(0, 0, 0, 255 - self.fade_alpha)
            painter.fillRect(self.rect(), fade_color)

    def update_wave(self):
        if not self.pixmap.isNull():  # 添加安全检查
            painter = QPainter(self.pixmap)
            self.pixmap.fill(Qt.black)
            painter.setPen(QPen(QColor(40, 40, 40), 1))
            self.draw_grid(painter)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            self.points = self.generate_ecg_wave(self.index)
            path = [QPointF(x, y) for x, y in self.points]
            for i in range(len(path) - 1):
                painter.drawLine(path[i], path[i + 1])
            self.index = (self.index + 1) % 200
            self.update()

    def draw_grid(self, painter):
        painter.setPen(QPen(QColor(40, 40, 40), 1))
        for x in range(0, self.width(), 20):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 20):
            painter.drawLine(0, y, self.width(), y)

    def generate_ecg_wave(self, offset):
        points = []
        for x in range(1000):
            t = (x + offset) / 40.0
            y = 360 - self.ecg_function(t) * 60
            points.append((x, y))
        return points

    def ecg_function(self, t):
        return (
                np.sin(2 * np.pi * t) * np.exp(-((t % 1) - 0.5) ** 2 / 0.01) +
                0.1 * np.sin(2 * np.pi * t * 5) +
                0.05 * np.sin(2 * np.pi * t * 1.5)
        )


def launch_animation():
    app = QApplication(sys.argv)
    window = ECGWidget()
    window.show()
    app.exec_()


if __name__ == "__main__":
    launch_animation()