import json
import matplotlib.pyplot as plt


def plot_paths(paths):
    i = 0
    for path in paths:
        plt.plot(path['x'], path['y'], '*')
        plt.text(path['x'][-1], path['y'][-1], str(i))
        i+=1
    

# 以二进制读取模式打开文件
with open('paths_json.bin', 'rb') as file:
    # 读取字节数据
    byte_data = file.read()
    # 将字节数据解码为 JSON 字符串
    json_str = byte_data.decode()
    # 使用 json.loads() 方法将 JSON 字符串解析为列表
    loaded_list = json.loads(json_str)
    # print(loaded_list)
    plot_paths(loaded_list)
    plt.axis('equal')
    plt.grid('both')
    plt.show()