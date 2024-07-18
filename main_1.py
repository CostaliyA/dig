import cv2
# import torch
from my_serial import MySerial
import threading
import queue
# from ultralytics import YOLO
import os
import numpy as np

# 控制机械臂位置
CONTROL_ROBOTIC_ARM_POSITION_DATA = "3001070155a11131"
# 控制机械臂抓取
CONTROL_ROBOTIC_ARM_GRAB_DATA = "3002070155a131"
# 单片机的功能选择
SINGLE_CHIP_FUNCTION_DATA = "3001070155a1ff31"
# 控制机械臂
CONTROL_ROBOTIC = "3001070155a151"

class Recognize:
    def __init__(self):
        # 初始化摄像头
        self.cap = cv2.VideoCapture('/dev/video8')
        # 初始化串口
        self.myserial = MySerial('/dev/ttyS4', baudrate=115200, timeout=1)
        self.myserial.send_msg(SINGLE_CHIP_FUNCTION_DATA)
        # 设置一个消息队列
        self.my_queue = queue.Queue(5)
        self.q_to_be_infered_img = queue.Queue(5)
        # 初始化模型
        # self.model = YOLO('./weights/best.pt')

        # 使用线程去不断接收数据
        t_recv_msg = threading.Thread(target=self.myserial.receive_msg)
        t_recv_msg.start()

        # 使用线程去不断处理数据
        t_msg_deal = threading.Thread(target=self.deal_msg)
        t_msg_deal.start()
        self.command_map = {
            'q': '30',  # 复位
            'a': '31',  # 向左
            'd': '32',  # 向右
            'w': '33',  # 向前
            's': '34',  # 向后
            'g': '37',  # 抓取
            'h': '38',  # 松开
            'z': '39',  # 点头
            'x': '3A',  # 摇头
            ' ': 'FF'  # 停止
        }
        self.command_map_f = {
            '30':'复位',
            '31':'向左',
            '32':'向右',
            '33':'向前',
            '34':'向后',
            '37':'抓取',
            '38':'松开',
            '39':'点头',
            '3A':'摇头',
            'FF':'停止'
        }
        # 初始化完毕
        print('初始化完毕')

    def deal_msg(self):
        while True:
            if self.myserial.recv_msg != '':
                if self.myserial.recv_msg[12:16] == '2131':
                    print('机械臂已到达仓库1')
                    self.my_queue.put('机械臂已到达仓库1')
                    self.myserial.recv_msg = ''
                elif self.myserial.recv_msg[12:16] == "4131":
                    print('机械臂已经搬运完毕')
                    self.my_queue.put('机械臂已经搬运完毕')
                    self.myserial.recv_msg = ''

    def control_arm(self):
        # 将机械臂移到仓库一的上面
        ls = ['21','22','23','24']
        cnt = 0
        while True:
            if cnt>=4:
                print("全部搬运完毕，程序退出")
                os._exit(0)
            print('将机械臂移到仓库一的上面')
            self.myserial.send_msg(CONTROL_ROBOTIC_ARM_POSITION_DATA)
            results = self.my_queue.get()
            if results == '机械臂已到达仓库1':
                print('准备读取照片')
                #摄像头被占用的异常检测，待补
                ret, frame = self.cap.read()
                print('准备识别照片')
                self.q_to_be_infered_img.put(frame)
                res = self.predict()
                min_value = float('inf')
                min_key = None
                flag = False
                for key, value in res.items():
                    flag = True
                    if value < min_value:
                        min_value = value
                        min_key = key
                if not flag:
                    print("全部搬运完毕，程序退出")
                    os._exit(0)
                if min_key == 0:
                    self.myserial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + '11' + ls[cnt])
                    # print('在左上角')
                elif min_key == 1:
                    self.myserial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + '12' + ls[cnt])
                    print('在右上角')
                elif min_key == 2:
                    self.myserial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + '13' + ls[cnt])
                    print('在左下角')
                elif min_key == 3:
                    self.myserial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + '14' + ls[cnt])
                    print('在右下角')
                cnt += 1
                # print(res)
                # x_min, y_min, x_max, y_max = self.recognize_pic(frame)
                # print('识别完毕，准备抓取')
                # if x_min <= 320 and y_min <= 160:
                #     self.myserial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + '1121')
                #     # print('在左上角')
                # elif x_min <= 320 and y_min > 160:
                #     self.myserial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + '1323')
                #     # print('在左下角')
                # elif x_min > 320 and y_min <= 160:
                #     self.myserial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + '1222')
                #     # print('在右上角')
                # elif x_min > 320 and y_min > 160:
                #     self.myserial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + '1424')
                #     # print('在右下角')
            # elif results == '机械臂已经搬运完毕':
            #     print('准备退出程序')
            #     os._exit(0)

    def on_press(self,key):
        try:
            key_char = key.char
            if key_char in self.command_map:
                command = self.command_map[key_char]
                print(self.command_map_f[command])
                self.myserial.send_msg(CONTROL_ROBOTIC + command)
        except AttributeError:
            if key == keyboard.Key.up:
                command = '35'  # 向上
                print("向上")
                self.myserial.send_msg(CONTROL_ROBOTIC + command)
            elif key == keyboard.Key.down:
                command = '36'  # 向下
                print("向下")
                self.myserial.send_msg(CONTROL_ROBOTIC + command)

    def on_release(self,key):
        if key == keyboard.Key.esc:
            return False

    def control_robot(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()
    def predict(self):
        height, width = 480, 640
        length = max(height, width)
        # 计算缩放比例，用于后续边界框坐标的缩放
        scale = length / 640
        model: cv2.dnn.Net = cv2.dnn.readNetFromONNX("./weights/best.onnx")
        names = "1;2;3;4;5;6;7;8;9".split(";")
        while True:
            img = self.q_to_be_infered_img.get()
            # 创建一个空白图像，尺寸为正方形，边长为较大的一边
            image = np.zeros((length, length, 3), np.uint8)
            # 将原始图像复制到空白图像中心
            image[0:height, 0:width] = img
            # 生成网络输入的blob
            blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
            # 将blob设置为模型的输入
            model.setInput(blob)
            # 前向传播，获取输出
            outputs = model.forward()

            # 显示结果
            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []
            output = outputs[0]
            for i in range(rows):
                # 提取每个边界框的类别概率
                classes_scores = output[i][4:]
                # 获取最大概率的类别索引和概率
                minScore, maxScore, minClassLoc, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
                # 如果概率大于等于0.8，则保存边界框信息
                if maxScore >= 0.8:
                    box = [output[i][0] - 0.5 * output[i][2], output[i][1] - 0.5 * output[i][3],
                           output[i][2], output[i][3]]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            # 使用非最大抑制（NMS）合并重叠的边界框
            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
            # 输出结果
            result = {}
            for index in result_boxes:
                box = boxes[index]
                box_out = [round(box[0] * scale), round(box[1] * scale),
                           round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale)]
                # print("矩形:", names[class_ids[index]], scores[index], box_out)
                # 计算矩形框的宽度和高度
                box_width = box_out[2] - box_out[0]
                box_height = box_out[3] - box_out[1]

                # 计算矩形框的面积
                area = box_width * box_height
                if area < 3000:
                    continue

                class_name = names[class_ids[index]]
                # score = scores[index]
                # 获取矩形框的中心坐标
                box_center_x = (box_out[0] + box_out[2]) / 2
                box_center_y = (box_out[1] + box_out[3]) / 2
                if (box_center_x < (width / 2)) and (box_center_y < (height / 2)):
                    result[0] = int(class_name)
                elif (box_center_x > (width / 2)) and (box_center_y < (height / 2)):
                    result[1] = int(class_name)
                elif (box_center_x < (width / 2)) and (box_center_y > (height / 2)):
                    result[2] = int(class_name)
                elif (box_center_x > (width / 2)) and (box_center_y > (height / 2)):
                    result[3] = int(class_name)
            return result

    # def recognize_pic(self, image):
    #     recognize_results = self.model(image)
    #     for result in recognize_results:
    #         datas = result.boxes.data
    #         if datas is not None:
    #             data_label = datas[:, -1]
    #             min_index = torch.argmin(data_label)
    #             x_min, y_min, x_max, y_max = datas[min_index][:4]
    #             return x_min, y_min, x_max, y_max

recognize = Recognize()
recognize.control_robot()
recognize.control_arm()
