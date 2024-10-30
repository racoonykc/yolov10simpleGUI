import os
import cv2
import numpy as np
import onnxruntime as ort
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6 import QtWidgets, QtCore
from postprocessing import postprocess
from preprocessing import preprocess

class VideoProcessingWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.classname =['fish']# 替换为实际的类别名称
        self.threshold = 0.5  # 初始化阈值
        self.init_gui()
        self.session = None
        self.timer = QtCore.QTimer()
        self.cap = None
        self.video_path = None
        self.output_path = None

    def init_gui(self):
        self.setWindowTitle('实时视频检测')
        self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(100, 100, 1600, 900)

        # 创建中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QtWidgets.QGridLayout(central_widget)
        main_layout.setSpacing(10)

        # 原始视频和处理后视频显示窗口
        self.oriVideoLabel = QtWidgets.QLabel(self)
        self.oriVideoLabel.setStyleSheet('border:1px solid #D7E2F9;')
        self.oriVideoLabel.setMinimumSize(400, 300)
        self.oriVideoLabel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.processedVideoLabel = QtWidgets.QLabel(self)
        self.processedVideoLabel.setStyleSheet('border:1px solid #D7E2F9;')
        self.processedVideoLabel.setMinimumSize(400, 300)
        self.processedVideoLabel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.imageDisplayLabel = QtWidgets.QLabel(self)
        self.imageDisplayLabel.setStyleSheet('border:1px solid #D7E2F9;')
        self.imageDisplayLabel.setMinimumSize(400, 300)
        self.imageDisplayLabel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.targetDisplayLabel = QtWidgets.QTextEdit(self)
        self.targetDisplayLabel.setStyleSheet('border:1px solid #D7E2F9;')
        self.targetDisplayLabel.setMinimumSize(400, 300)
        self.targetDisplayLabel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.targetDisplayLabel.setReadOnly(True)

        # 将窗口放入网格布局中
        main_layout.addWidget(self.oriVideoLabel, 0, 0)
        main_layout.addWidget(self.processedVideoLabel, 0, 1)
        main_layout.addWidget(self.imageDisplayLabel, 1, 0)
        main_layout.addWidget(self.targetDisplayLabel, 1, 1)

        # 添加控制按钮
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(15)
        self.selectModelBtn = QtWidgets.QPushButton('📂选择模型')
        self.selectModelBtn.setFixedSize(120, 50)
        self.selectModelBtn.clicked.connect(self.load_model)
        self.openVideoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.openVideoBtn.setFixedSize(120, 50)
        self.openVideoBtn.clicked.connect(self.start_video)
        self.openVideoBtn.setEnabled(False)
        self.stopDetectBtn = QtWidgets.QPushButton('🛑停止')
        self.stopDetectBtn.setFixedSize(120, 50)
        self.stopDetectBtn.setEnabled(False)
        self.stopDetectBtn.clicked.connect(self.stop_detect)
        self.exitBtn = QtWidgets.QPushButton('⏹退出')
        self.exitBtn.setFixedSize(120, 50)
        self.exitBtn.clicked.connect(self.close)

        control_layout.addWidget(self.selectModelBtn)
        control_layout.addWidget(self.openVideoBtn)
        control_layout.addWidget(self.stopDetectBtn)
        control_layout.addWidget(self.exitBtn)

        main_layout.addLayout(control_layout, 2, 0, 1, 2)

    def load_model(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选取模型权重", filter='*.onnx')
        if file_name.endswith('.onnx'):
            self.session = ort.InferenceSession(file_name)
            self.statusBar().showMessage(f"已加载模型: {file_name}")
        else:
            self.statusBar().showMessage("重新选择模型")

        self.openVideoBtn.setEnabled(True)
        self.stopDetectBtn.setEnabled(True)
        
    def start_video(self):
        if self.timer.isActive():
            self.timer.stop()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选取视频文件", filter='*.mp4')
        if os.path.isfile(file_name):
            self.video_path = file_name
            self.statusBar().showMessage(f"已选择视频文件: {file_name}")
            self.cap = cv2.VideoCapture(self.video_path)
            self.timer.timeout.connect(self.process_video)
            self.timer.start(30)
        else:
            self.statusBar().showMessage("重新选择视频文件")

    def process_video(self):
        ret, frame = self.cap.read()
        if ret:
            original_h, original_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_qimage = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QPixmap.fromImage(frame_qimage))
            self.oriVideoLabel.setScaledContents(True)

            if self.session is not None:
                input_tensor, ratio, x_offset, y_offset = preprocess(frame, original_w, original_h, 640, 640)
                input_name = self.session.get_inputs()[0].name
                output_name = self.session.get_outputs()[0].name
                outputs = self.session.run([output_name], {input_name: input_tensor})[0]
                output = np.squeeze(outputs)
                processed_frame = postprocess(output, frame, ratio, x_offset, y_offset,self.classname, self.threshold)
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                processed_frame_qimage = QImage(processed_frame_rgb.data, processed_frame_rgb.shape[1], processed_frame_rgb.shape[0], QImage.Format_RGB888)
                self.processedVideoLabel.setPixmap(QPixmap.fromImage(processed_frame_qimage))
                self.processedVideoLabel.setScaledContents(True)

                # 绘制目标信息并显示
                image = self.draw_targets(output, ratio, x_offset, y_offset, original_w, original_h)
                image_qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                self.imageDisplayLabel.setPixmap(QPixmap.fromImage(image_qimage))
                self.imageDisplayLabel.setScaledContents(True)

                # 显示目标信息
                target_text = self.get_target_text(output, ratio, x_offset, y_offset, original_w, original_h)
                self.targetDisplayLabel.setPlainText(target_text)

    def draw_targets(self, output, ratio, x_offset, y_offset, src_w, src_h, confidence_threshold=0.5):
        # 创建一个空白图像
        image_height, image_width = src_h, src_w
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        for i in range(output.shape[0]):
            confidence = float(output[i][4])  # 确保 confidence 是一个标量值
            if confidence > confidence_threshold:
                label = int(output[i][5])
                xmin = int((output[i][0] - x_offset) / ratio)
                ymin = int((output[i][1] - y_offset) / ratio)
                xmax = int((output[i][2] - x_offset) / ratio)
                ymax = int((output[i][3] - y_offset) / ratio)

                # 绘制矩形框
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # 标注目标ID
                cv2.putText(image, f'ID: {i}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    def get_target_text(self, output, ratio, x_offset, y_offset, src_w, src_h, confidence_threshold=0.5):
        # 获取目标文本信息
        target_info = []
        for i in range(output.shape[0]):
            confidence = float(output[i][4])  # 确保 confidence 是一个标量值
            if confidence > confidence_threshold:
                label = int(output[i][5])
                xmin = int((output[i][0] - x_offset) / ratio)
                ymin = int((output[i][1] - y_offset) / ratio)
                xmax = int((output[i][2] - x_offset) / ratio)
                ymax = int((output[i][3] - y_offset) / ratio)

                target_info.append(f'ID: {i}, Confidence: {confidence:.2f}, Box: ({xmin}, {ymin}), ({xmax}, {ymax})')

        return '\n'.join(target_info)

    def stop_detect(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_path = None
        img = cv2.cvtColor(np.zeros((500, 500), np.uint8), cv2.COLOR_BGR2RGB)
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.oriVideoLabel.setPixmap(QPixmap.fromImage(img))
        self.processedVideoLabel.setPixmap(QPixmap.fromImage(img))
        self.imageDisplayLabel.setPixmap(QPixmap.fromImage(img))
        self.targetDisplayLabel.setPlainText("")
        self.statusBar().showMessage("检测已停止")

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.timer.isActive():
            self.timer.stop()
        if self.socket:
            self.socket.close()
        self.statusBar().showMessage("应用程序已退出")
        exit()
