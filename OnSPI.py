# OnSPI.py

import spidev
import time
import threading

class ADC128S102:
    """
    ADC128S102 SPI ADC 控制类。
    通过 SPI 接口读取 ADC128S102 芯片的模拟电压值，并转换为电压 (V)。
    """
    def __init__(self, bus=0, device=0, speed=1000000, vref=3.3):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed
        self.spi.mode = 0b00  # SPI 模式 0
        self.vref = vref

    def read_data(self):
        """
        发送 2 字节占位后读取 12 位原始值，并将其转换为电压值 (V)。
        """
        tx = [0x00, 0x00]
        rx = self.spi.xfer2(tx)
        raw = ((rx[0] & 0x0F) << 8) | rx[1]
        return raw * self.vref / 4096.0

    def close(self):
        try:
            self.spi.close()
        except:
            pass


class SPIDataCollector:
    """
    通过 SPI 实时采集 ECG 电压数据。
    采样率设为 250 Hz，每 4 个点打包一次推到 data_queue，并附带最后一个点的真实 time.time() 时间戳。
    """
    def __init__(self):
        try:
            self.adc = ADC128S102()
        except Exception as e:
            raise RuntimeError(f"SPIDataCollector: 初始化 ADC128S102 失败: {e}")

        # 目标采样率 250 Hz
        self.sampling_rate = 250
        self.sample_interval = 1.0 / self.sampling_rate  # 4 ms 理论
        self.batch_size = 4
        self.running = False

        # 直流偏置值（请根据实测修改，此处设置为 0.6V）
        self.dc_offset = 1.1

        self.thread = None

    def start(self, data_queue):
        """
        在单独线程中调用 start_stream，以普通线程优先级运行。
        """
        self.thread = threading.Thread(
            target=self.start_stream,
            args=(data_queue,),
            daemon=True
        )
        self.thread.start()
        print(f"SPIDataCollector: 采样线程 (TID={self.thread.native_id}) 启动，采用普通优先级")

    def start_stream(self, data_queue):
        self.running = True
        voltage_buffer = []
        time_buffer = []

        # 用 monotonic() 做精确调度
        next_t = time.monotonic()
        while self.running:
            # 1. 读取一次 ADC
            try:
                voltage = self.adc.read_data()
            except Exception as e:
                print(f"SPIDataCollector: 读取 ADC 失败: {e}")
                time.sleep(self.sample_interval)
                continue

            t_now = time.time()  # 真实系统时间戳，用于后续绘图/记录 CSV
            voltage_buffer.append(voltage)
            time_buffer.append(t_now)

            # 2. 如果积累到 batch_size 个点，就打包推送
            if len(voltage_buffer) >= self.batch_size:
                packet_time = time_buffer[-1]
                # 去除直流偏置后放大 3 倍
                adjusted = [ (v - self.dc_offset) * 3 for v in voltage_buffer ]
                try:
                    data_queue.put((packet_time, adjusted))
                except Exception as e:
                    print(f"SPIDataCollector: 推送到队列失败: {e}")
                voltage_buffer.clear()
                time_buffer.clear()

            # 3. 精确等待到下一次采样时刻
            next_t += self.sample_interval
            sleep_dur = next_t - time.monotonic()
            if sleep_dur > 0:
                time.sleep(sleep_dur)
            else:
                # 如果已经来不及，就重置 next_t，继续下一次循环
                next_t = time.monotonic()

        # 停止时发结束信号
        try:
            data_queue.put((None, None))
        except:
            pass

        # 关闭 ADC
        try:
            self.adc.close()
        except:
            pass

    def stop(self):
        """停止采集线程"""
        self.running = False
        print("SPIDataCollector: 停止采集")