# -*- coding: utf-8 -*-

'''
Author: fanyu fantiming@yeah.net
Date: 2025-03-27 11:35:29
LastEditors: fanyu fantiming@yeah.net
LastEditTime: 2025-03-27 12:06:54
FilePath: /DTPP/debug/plot_times.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%A
'''


import json
import matplotlib.pyplot as plt


# 以二进制读取模式打开文件
with open('time_list.bin', 'rb') as file:
    # 读取字节数据
    byte_data = file.read()
    # 将字节数据解码为 JSON 字符串
    json_str = byte_data.decode()
    # 使用 json.loads() 方法将 JSON 字符串解析为列表
    loaded_list = json.loads(json_str)
    fig = plt.figure()
    t1 = [t['r1'] for t in loaded_list]
    p1 = [t['p1'] for t in loaded_list]
    t2 = [t['r2'] for t in loaded_list]
    p2 = [t['p2'] for t in loaded_list]
    cs = [t['cs'] for t in loaded_list]

    frame = range(len(t1))
    plt.plot(frame, t1, 'r', label='t1')
    plt.plot(frame, p1, 'g', label='p1')
    plt.plot(frame, t2, 'b', label='t2')
    plt.plot(frame, p2, 'y', label='p2')
    plt.plot(frame, cs, 'c', label='cs')
    plt.legend()

    # 设置 x 轴范围
    
    plt.xlim(0, len(t1))
    # 显示x，y轴范围
    y_max = max(max(t1), max(p1), max(t2), max(p2), max(cs))
    plt.ylim(0, y_max*1.1)
    # plt.axis('equal')
    # 设置 x 轴标签
    plt.xlabel('frame')
    # 设置 y 轴标签
    plt.ylabel('time/ms')
    # 设置标题
    plt.title('Time')
    # 设置刻度
    plt.xticks(range(0, len(t1), 10))
    plt.yticks(range(0, int(y_max*1.1), 20))
    # 显示网格
    plt.grid('both')
    plt.show()