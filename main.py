'''
HO CHI MINH CITY UNIVERSITY OF TECHNOLOGY
       NGUYEN THANH DAT - 1510698
LVTN: Nhận dạng ngôn ngữ ký hiệu cho người khiếm thính sử dụng kỹ thuật học sâu:
Tách và phân tích đặc trưng khung xương trên video RGB
'''
"""
guthub: https://github.com/Dreamer179
email: thanhdatbku97@gmail.com
"""
# -*- coding: UTF-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets

import time
import settings
import numpy as np
import cv2
import sys, os, time, argparse, logging
import simplejson
import argparse
import mylib.io as myio
from mylib.display import drawActionResult
from mylib.data_preprocessing import pose_normalization_20
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common

from pathlib import Path
from Tracking.deep_sort import preprocessing
from Tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from Tracking.deep_sort.detection import Detection
from Tracking import generate_dets as gdet
from Tracking.deep_sort.tracker import Tracker

# Use Deep-sort(Simple Online and Realtime Tracking)
# To track multi-person for multi-person actions recognition

# Xác định các tham số cơ bản
file_path = Path.cwd()
clip_length = 15
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

#deep_sort
model_filename = str(file_path/'Tracking/graph_model/mars-small128.pb')
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# track_box
trk_clr = (0, 255, 0)


class SkeletonDetector(object):
    def __init__(self, model="mobilenet_thin"):
        models = set({"mobilenet_thin", "cmu", "VGG_origin", "mobilenet_v2_large" , "mobilenet_v2_small"})
        self.model = model if model in models else "mobilenet_thin"
        self.resize_out_ratio = 4.0
        w, h = model_wh("432x368")
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(self.model),
                                target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))
        self.w, self.h = w, h
        self.e = e
        self.fps_time = time.time()

    def detect(self, image):
        t = time.time()

        # Inference
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                #   upsample_size=self.args.resize_out_ratio)
                                  upsample_size=self.resize_out_ratio)


        return humans
    
    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
        self.fps_time = time.time()

    @staticmethod
    def humans_to_skelsInfo(humans, joint_choose, action_type="None"):
        # skeleton = [action_type, 18*[x,y], 18*score]
        skelsInfo, skelsInfo_choose, bboxes = [], [], []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(1+18*2+18)
            skeleton[0] = action_type
            skel = []
            lx = []
            ly = []
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[1+2*idx]=body_part.x
                lx.append(int(body_part.x * 960 + 0.5)) # 960: width of frame (chon tuy theo camera)
                skeleton[1+2*idx+1]=body_part.y
                ly.append(int(body_part.y * 720 + 0.5)) # 720: height of frame (chon tuy theo camera)
                if i in joint_choose:
                    skel.append(body_part.x)
                    skel.append(body_part.y)
                # skeleton[1+36+idx]=body_part.score
            #find bouding box of a SJM
            bboxes.append([min(lx), min(ly), max(lx)-min(lx), max(ly)-min(ly)])
            skelsInfo.append(skeleton)
            skelsInfo_choose.append(skel)
        return skelsInfo, skelsInfo_choose, bboxes

    def humans_to_skelsInfo_choose(humans, joint_choose):
        # skeleton = [action_type, 18*[x,y], 18*score]
        skelsInfo_choose = []
        for human in humans:
            skeleton_choose = []
            for i, body_part in human.body_parts.items(): # iterate dict
                #idx = body_part.part_idx
                for element in joint_choose:
                    if i == element:
                        skeleton_choose.append(body_part.x)
                        skeleton_choose.append(body_part.y)

                print('i: ', i)
                print('body_part.x: ', body_part.x)
                print('body_part.y: ', body_part.y)
            skelsInfo_choose.append(skeleton_choose)
                # skeleton[1+36+idx]=body_part.score
        return skelsInfo_choose
    
    @staticmethod
    def get_ith_skeleton(skelsInfo, ith_skeleton=0):
        return np.array(skelsInfo[ith_skeleton][1:1+18*2])

    def get_ith_skeleton_choose(skelsInfo, ith_skeleton=0):
        return np.array(skelsInfo[ith_skeleton])

class ActionClassifier(object):
    
    def __init__(self, model_path):
        from keras.models import load_model

        self.dnn_model = load_model(model_path)
        self.action_dict = ['xin chao', 'toi', 'thanh pho', 'vui ve', 'am em', 'Sai Gon', 'di bo', 'mua mang', 'doi bung', 'yeu', 'an', 'bieu quyet', 'dung yen', 'hep', 'rong', 'Vinh Long']
        self.vni_action_dict = [u'xin chào', u'Tôi', u'thành phố', u'vui vẻ', u'ẵm em', u'Sài Gòn', u'đi bộ', u'mùa màng', u'đói bụng', u'yêu', u'ăn', u'biểu quyết', u'.', u'hẹp', u'rộng', u'Vĩnh Long']
    def predict(self, skeleton):

        # Preprocess data
        tmp = pose_normalization_20(skeleton)# tmp = skeleton # ko normalize
        skeleton_input = np.array(tmp).reshape(-1, len(tmp))
            
        # Predicted label: int & string
        kq = self.dnn_model.predict(skeleton_input)
        predicted_idx = np.argmax(kq) #skeleton_input
        acc = np.max(kq)
        prediced_label = self.action_dict[predicted_idx]
        vni_prediced_label = self.vni_action_dict[predicted_idx]

        return prediced_label, acc, vni_prediced_label
############################################################################################################################################

class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        #self.tracker = Sort(settings.sort_max_age, settings.sort_min_hit)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture(0)
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_mode = 0
        self.fps = 0.00
        self.data = {}
        self.memory = {}
        self.joints = []
        self.current = []
        self.previous = []
        self.text = ''
        self.dict = {}
        self.dict_check = {}

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'Camera OFF') 
        self.button_mode_1 = QtWidgets.QPushButton(u'Skeleton OFF') 
        self.button_mode_2 = QtWidgets.QPushButton(u'Tracking OFF')
        self.button_mode_3 = QtWidgets.QPushButton(u'Recognition OFF')
        self.button_close = QtWidgets.QPushButton(u'Close')

        self.button_open_camera.setMinimumHeight(50)
        self.button_mode_1.setMinimumHeight(50)
        self.button_mode_2.setMinimumHeight(50)
        self.button_mode_3.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        self.button_close.move(10, 100)

        self.infoBox = QtWidgets.QTextBrowser(self)
        self.infoBox.setGeometry(QtCore.QRect(10, 320, 400, 90))

        self.textBox = QtWidgets.QTextBrowser(self) #####################################
        self.textBox.setGeometry(QtCore.QRect(10, 420, 400, 310)) #########################
        #self.textBox.setPointSize(16)

        self.font = QtGui.QFont()
        self.font.setPointSize(26)
        self.textBox.setFont(self.font)

        # show info
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(400, 400) #(200, 200)

        self.label_show_camera.setFixedSize(settings.winWidth + 1, settings.winHeight + 1)
        self.label_show_camera.setAutoFillBackground(True)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_mode_1)
        self.__layout_fun_button.addWidget(self.button_mode_2)
        self.__layout_fun_button.addWidget(self.button_mode_3)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'Real-time multi-person Vietnamese sign language recognition system')

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_event)
        self.timer_camera.timeout.connect(self.show_camera)

        self.button_mode_1.clicked.connect(self.button_event)
        self.button_mode_2.clicked.connect(self.button_event)
        self.button_mode_3.clicked.connect(self.button_event)
        self.button_close.clicked.connect(self.close)

    def button_event(self):
        sender = self.sender()
        if sender == self.button_mode_1 : #and self.timer_camera.isActive()
            if self.__flag_mode != 1:
                self.__flag_mode = 1
                self.button_mode_1.setText(u'Skeleton ON') 
                self.button_mode_2.setText(u'Tracking OFF')
                self.button_mode_3.setText(u'Recognition OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_1.setText(u'Skeleton OFF')
                self.infoBox.setText(u'Camera is on')
        elif sender == self.button_mode_2 : #and self.timer_camera.isActive()
            if self.__flag_mode != 2:
                self.__flag_mode = 2
                self.button_mode_1.setText(u'Skeleton OFF')
                self.button_mode_2.setText(u'Tracking ON')
                self.button_mode_3.setText(u'Recognition OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_2.setText(u'Tracking OFF')
                self.infoBox.setText(u'Camera is on')
        elif sender == self.button_mode_3 : #and self.timer_camera.isActive()
            if self.__flag_mode != 3:
                self.__flag_mode = 3
                self.button_mode_1.setText(u'Skeleton OFF')
                self.button_mode_2.setText(u'Tracking OFF')
                self.button_mode_3.setText(u'Recognition ON')
            else:
                self.__flag_mode = 0
                self.button_mode_3.setText(u'Recognition OFF')
                self.infoBox.setText(u'Camera is on')
        else:
            self.__flag_mode = 0
            self.button_mode_1.setText(u'Skeleton OFF')
            self.button_mode_2.setText(u'Tracking OFF')
            self.button_mode_3.setText(u'Recognition OFF')
            if self.timer_camera.isActive() == False:
                flag = self.cap.open(self.CAM_NUM)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.winWidth)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.winHeight)
                if flag == False:
                    msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check if the camera and computer are connected correctly.", #请检测相机与电脑是否连接正确
                                                        buttons=QtWidgets.QMessageBox.Ok,
                                                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_camera.start(1)
                    self.button_open_camera.setText(u'Camera ON')
                    self.infoBox.setText(u'Camera is on')
            else:
                self.timer_camera.stop()
                self.cap.release()
                self.label_show_camera.clear()
                self.button_open_camera.setText(u'Camera OFF')
                self.infoBox.setText(u'Camera is off')

    def show_camera(self):
        start = time.time()
        ret, frame = self.cap.read()
        # image_h, image_w = frame.shape[:2] # (720x960)
        #show = cv2.resize(frame, (settings.winWidth, settings.winHeight))
        show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            if self.__flag_mode == 1:
                self.infoBox.setText(u'Pose estimation')
                humans = my_detector.detect(show)
                #skelsInfo = SkeletonDetector.humans_to_skelsInfo(humans)
                #print('SkelsInfo: ', skelsInfo)
                my_detector.draw(show, humans)
                # for ith_skel in range(0, len(skelsInfo)):
                # 	skeleton = SkeletonDetector.get_ith_skeleton(skelsInfo, ith_skel)
                # 	if ith_skel:#ith_skel == 0:
                	   

            elif self.__flag_mode == 2:
                self.infoBox.setText(u'Multiplayer tracking')
                prediced_label = ''
                humans = my_detector.detect(show)
                my_detector.draw(show, humans)
                skelsInfo, skelsInfo_choose, bboxes = SkeletonDetector.humans_to_skelsInfo(humans, joint_choose)
                if bboxes:
                    bboxes = np.array(bboxes)
                    features = encoder(frame, bboxes)

                    # score to 1.0 here
                    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]

                    # Non-maximal suppression 进行非极大抑制
                    boxes = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                    detections = [detections[i] for i in indices]

                    # Call tracker and update in real time 调用tracker并实时更新
                    tracker.predict()
                    tracker.update(detections)

                    # Record track results, including bounding boxes and their ID 记录track的结果，包括bounding boxes及其ID
                    trk_result = []
                    #trk.track_id = 1
                    for trk in tracker.tracks:
                        if not trk.is_confirmed() or trk.time_since_update > 1:
                            continue
                        bbox = trk.to_tlwh()
                        print('bbox: ', bbox)
                        trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
                        # Đánh dấu track_ID
                        trk_id = 'ID-' + str(trk.track_id)
                        print('track ID: ', trk_id)
                        cv2.putText(show, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)################################
                    for d in trk_result: # gom nhieu bboxes
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2]) + xmin
                        ymax = int(d[3]) + ymin
                        label = int(d[4])
                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)



            elif self.__flag_mode == 3:
                self.infoBox.setText(u'Sign language recognition')
                humans = my_detector.detect(show)
                my_detector.draw(show, humans)
                #skelsInfo_choose = SkeletonDetector.humans_to_skelsInfo_choose(humans, joint_choose) # cac skeleton_choose
                skelsInfo, skelsInfo_choose, bboxes = SkeletonDetector.humans_to_skelsInfo(humans, joint_choose) # cac skeleton_full
                # if len(skelsInfo) == 0:
                #     self.text = ''
                if bboxes:
                    bboxes = np.array(bboxes)
                    features = encoder(frame, bboxes)

                    # score to 1.0 here
                    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]

                    # Non-maximal suppression 
                    boxes = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                    detections = [detections[i] for i in indices]

                    # Call tracker and update in real time 
                    tracker.predict()
                    tracker.update(detections)

                    # Record track results, including bounding boxes and their ID 
                    trk_result = []
                    #trk.track_id = 1
                    for trk in tracker.tracks:
                        if not trk.is_confirmed() or trk.time_since_update > 1:
                            continue
                        bbox = trk.to_tlwh()
                        #print('bbox: ', bbox)
                        trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
                        # Đánh dấu track_ID
                        trk_id = 'ID-' + str(trk.track_id)
                        #print('track ID: ', trk_id)
                        cv2.putText(show, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)################################
                    list_label = []
                    for d in trk_result: # gom nhieu bboxes
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2]) + xmin
                        ymax = int(d[3]) + ymin
                        label = int(d[4])
                        list_label.append(label)
                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)
                        try:
                            # xcenter là giá trị tọa độ x của tất cả các khớp (cổ) ​​của con người trong một khung hình ảnh
                            # Khớp ID bằng cách tính khoảng cách giữa track_box và xcenter của con người
                            tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
                            j = np.argmin(tmp)
                        except:
                            # Nếu không có người trong khung hiện tại, mặc định j = 0 (không hợp lệ)
                            j = 0

                        if skelsInfo_choose:
                            skeleton_choose = skelsInfo_choose[j]
                            skeleton = skelsInfo[j]
                            skeleton = np.array(skeleton[1:1+18*2])
                            #skeleton = 
                            if len(skeleton_choose) == 20:
                                #skeleton_choose = np.array(skelsInfo_choose).reshape(-1,20)
                                prediced_label, acc, vni_prediced_label = classifier.predict(skeleton_choose)
                                if acc > 0.94:
                                    if label not in list(self.dict):
                                        self.dict[label] = vni_prediced_label
                                        self.dict_check[label] = vni_prediced_label
                                    if label in list(self.dict) and (vni_prediced_label != self.dict_check[label]):
                                        self.dict[label] = self.dict[label] + ' ' + vni_prediced_label
                                        self.dict_check[label] = vni_prediced_label
                                    #print('prediced_label: ', prediced_label)
                                    cv2.putText(show, prediced_label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4, cv2.LINE_AA)
                    if len(self.dict_check) > 0:
                        for i in list(self.dict_check):
                            if i not in list_label:
                                del self.dict_check[i]
                                del self.dict[i]
                    self.text = ''
                    for i in list(self.dict):
                        self.text = self.text + 'ID-' + str(i) + ': ' + self.dict[i] + '\n'
                    self.textBox.setText(self.text)


            end = time.time()
            self.fps = 1. / (end - start)
            cv2.putText(show, 'FPS: %.2f' % self.fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            showImage = QtGui.QImage(show, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"Shut down", u"Whether it is closed!")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'OK')
        cancel.setText(u'cancel')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
            print("System exited.")


if __name__ == '__main__':
    my_detector = SkeletonDetector()
    classifier = ActionClassifier('model/sign_language_16.h5')
    joint_choose = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    print("Load all models done!")
    print("The system starts ro run.")
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
