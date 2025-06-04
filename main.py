import matplotlib
matplotlib.use('TkAgg')
import LaunchAnimation
LaunchAnimation.launch_animation()
import os
from tkinter import Tk, filedialog



def select_csv_file():
    root = Tk()
    root.withdraw()  # 不显示主窗口
    file_path = filedialog.askopenfilename(
        title="选择 CSV 文件",
        filetypes=[("CSV 文件", "*.csv")],
        initialdir=os.getcwd()
    )
    return file_path


import Platform

if __name__ == "__main__":
    Platform.launch_platform()