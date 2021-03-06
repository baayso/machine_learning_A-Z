#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May  4 10:26:25 2018

@author: http://www.cnblogs.com/neo-T/p/6426029.html
"""

import cv2
import sys
from PIL import Image


def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        # 显示图像并等待10毫秒按键输入，输入‘q’退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo("截取视频流", int(sys.argv[1]))
