import cv2
import numpy as np

def preprocess(bgr_image, src_w, src_h, dst_w, dst_h):
    # 转换颜色格式
    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # 计算缩放比例和边界
    ratio = min(dst_w/src_w, dst_h/src_h)
    border_w = int(round(src_w * ratio / 2) * 2)
    border_h = int(round(src_h * ratio / 2) * 2)
    x_offset = (dst_w - border_w) // 2  
    y_offset = (dst_h - border_h) // 2

    # 调整图像大小和边界
    image = cv2.resize(image, (border_w, border_h))
    image = cv2.copyMakeBorder(
        image, y_offset, y_offset, x_offset, x_offset, 
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # 归一化并调整维度
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    input_tensor = np.expand_dims(image, axis=0)
    return input_tensor, ratio, x_offset, y_offset
