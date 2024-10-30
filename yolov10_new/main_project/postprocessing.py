import cv2
import numpy as np

def postprocess(output, image, ratio, x_offset, y_offset,classname, confidence_threshold=0.2):
    CLASSES =classname  # 替换为实际的类别名称
    COLOR_LIST = [(0, 0, 1)]  # 替换为实际的颜色列表

    # 创建图像的深拷贝
    temp_image = image.copy()

    for i in range(output.shape[0]):
        confidence = float(output[i][4])  # 确保 confidence 是一个标量值
        if confidence > confidence_threshold:
            label = int(output[i][5])
            xmin = int((output[i][0] - x_offset) / ratio)
            ymin = int((output[i][1] - y_offset) / ratio)
            xmax = int((output[i][2] - x_offset) / ratio)
            ymax = int((output[i][3] - y_offset) / ratio)

            class_name = CLASSES[label]
            box_color = np.array(COLOR_LIST[label]) * 255
            box_color = (int(box_color[0]), int(box_color[1]), int(box_color[2]))
            cv2.rectangle(temp_image, (xmin, ymin), (xmax, ymax), box_color, 2)
            cv2.putText(temp_image, f'{class_name}: {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    # 恢复到原始图像大小
    original_h, original_w = temp_image.shape[:2]
    temp_image = cv2.resize(temp_image, (original_w, original_h))
    return temp_image

def draw_targets(output, ratio, x_offset, y_offset, src_w, src_h, confidence_threshold=0.5):
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

def get_target_text(output, ratio, x_offset, y_offset, src_w, src_h, confidence_threshold=0.5):
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
