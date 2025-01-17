###
 # Copyright (c) 2025 by GAC R&D Center, All Rights Reserved.
 # @Author: 范雨
 # @Date: 2025-01-10 10:34:38
 # @LastEditTime: 2025-01-13 10:21:34
 # @LastEditors: 范雨
 # @Description: 
### 

export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"

echo "这是一个例子，如下方式映射到你的conda env 中，下面例子为系统环境
例如:\n
# python 3.8 also works well, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.7/site-packages/carla.pth 

"

# 绿色打印提示信息
# 使用printf替代echo以支持转义序列
printf "\033[32m  --- CARLA_ROOT=\"%s\"\n \033[0m" "$CARLA_ROOT"

# # 将CARLA的egg文件路径添加到conda环境的site-packages目录
# printf "%s\n" "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> "$CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.7/site-packages/carla.pth"

# 打印conda路径信息
printf "\033[32m --- CONDA_PATH=\"%s\"\n \033[0m" "$CONDA_PATH"
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> "$CONDA_PATH"/envs/dtpp/lib/python3.9/site-packages/carla.pth

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla

# ./CarlaUE4.sh Carla/Maps/Town10HD_Opt -windowed -carla-server -benchmark -quality-level=Low -fps=30 -RenderOffScree
