
# YOLOv10 简易图形化界面

## 文件目录

- **GUI_all_in_one.py**：一个整合版的 GUI 界面，点击运行。
- **main_project**：分模块的 GUI 界面，内部结构如下：
  - **main.py**：主程序入口。
  - **preprocessing.py**：预处理模块，一般不用动。
  - **postprocessing.py**：后处理模块，可以在这里实现自己想要的功能。
  - **gui.py**：图形化界面，在这里更改功能。
- **weights**:我提供的权重存放处，下面有fish和mine两类，mine文件夹中有添加抑制训练数据和未抑制训练数据两种权重

## 环境配置

### 使用 Conda 创建环境

1. **安装 Anaconda**（如果尚未安装）：
   - 下载 Anaconda 安装包并进行安装。可以从 [Anaconda 官网](https://www.anaconda.com/products/distribution#download-section) 下载。

2. **打开 Anaconda Prompt**：
   - 在 Windows 上搜索 "Anaconda Prompt"，在 macOS 上打开终端。

3. **创建新的 Conda 环境**：
   ```bash
   conda create -n yolov10_env python=3.8
   ```

4. **激活新创建的环境**：
   ```bash
   conda activate yolov10_env
   ```

5. **安装所需库**：
   ```bash
   conda install opencv numpy onnxruntime
   pip install PySide6
   ```

6. **确保安装所有依赖**：
   - 如果有其他依赖，请在此处添加。
   
## 运行程序

1. **整合版 GUI**：
   - 运行整合版的 GUI：
   ```bash
   cd your_path
   conda activate yolov10_env
   python GUI_all_in_one.py
   ```

2. **分模块 GUI**：
   - 在 `main_project` 文件夹中运行主程序：
   ```bash
   cd main_project
   python main.py
   ```

## 注意事项


- 如果需要修改后处理模块或图形化界面，打开相应的 `.py` 文件进行修改。
- 有事建议先问gpt
- 如果模型容易把不是目标的物体识别出来，建议添加一些负面抑制数据：即随机图片和空标签
