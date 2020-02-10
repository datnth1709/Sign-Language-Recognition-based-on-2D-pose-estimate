# -*- coding: UTF-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
# from pose.estimator import TfPoseEstimator
# from pose.networks import get_graph_path
# from utils.sort import Sort
# from utils.actions import actionPredictor
# from utils.joint_preprocess import *
# import sys
# import cv2
# import numpy as np
import time
import settings
# from keras.models import load_model
# import mylib.io as myio
# from mylib.display import drawActionResult
# from mylib.data_preprocessing import pose_normalization
# import pyautogui as Gui
# poseEstimator = None
import pyautogui as Gui
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

# def load_all_model():
#     global poseEstimator, dnn_model
#     poseEstimator = TfPoseEstimator(
#         get_graph_path('VGG_origin'), target_size=(432, 368)) # mobilenet_v2_large , mobilenet_v2_small , mobilenet_thin , cmu, VGG_origin
#     dnn_model = load_model('model/sign_language_16.h5')


########################################################################################################################################
class SkeletonDetector(object):
    # This func is copied from https://github.com/ildoonet/tf-pose-estimation

    def __init__(self, model="mobilenet_v2_large"):
        models = set({"mobilenet_thin", "cmu", "VGG_origin", "mobilenet_v2_large" , "mobilenet_v2_small"})
        self.model = model if model in models else "mobilenet_thin"
        # parser = argparse.ArgumentParser(description='tf-pose-estimation run')
        # parser.add_argument('--image', type=str, default='./images/p1.jpg')
        # parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

        # parser.add_argument('--resize', type=str, default='0x0',
        #                     help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        # parser.add_argument('--resize-out-ratio', type=float, default=4.0,
        #                     help='if provided, resize heatmaps before they are post-processed. default=1.0')
        self.resize_out_ratio = 4.0

        # args = parser.parse_args()

        # w, h = model_wh(args.resize)
        w, h = model_wh("432x368")
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(self.model),
                                target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))

        # self.args = args
        self.w, self.h = w, h
        self.e = e
        self.fps_time = time.time()

    def detect(self, image):
        t = time.time()

        # Inference
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                #   upsample_size=self.args.resize_out_ratio)
                                  upsample_size=self.resize_out_ratio)

        # Print result and time cost
        print("humans:", humans)

        return humans
    
    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
        self.fps_time = time.time()

    @staticmethod
    def humans_to_skelsInfo(humans, action_type="None"):
        # skeleton = [action_type, 18*[x,y], 18*score]
        skelsInfo = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(1+18*2+18)
            skeleton[0] = action_type
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[1+2*idx]=body_part.x
                skeleton[1+2*idx+1]=body_part.y
                # skeleton[1+36+idx]=body_part.score
            skelsInfo.append(skeleton)
        return skelsInfo

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

    def predict(self, skeleton):

        # Preprocess data
        tmp = pose_normalization_20(skeleton)# tmp = skeleton # ko normalize
        skeleton_input = np.array(tmp).reshape(-1, len(tmp))
            
        # Predicted label: int & string
        kq = self.dnn_model.predict(skeleton_input)
        predicted_idx = np.argmax(kq) #skeleton_input
        acc = np.max(kq)
        prediced_label = self.action_dict[predicted_idx]

        return prediced_label, acc
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
        self.flag = ''

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
        #show = cv2.resize(frame, (settings.winWidth, settings.winHeight))
        show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            if self.__flag_mode == 1:
                self.infoBox.setText(u'Pose estimation')
                humans = my_detector.detect(show)
                skelsInfo = SkeletonDetector.humans_to_skelsInfo(humans)
                print('SkelsInfo: ', skelsInfo)
                for ith_skel in range(0, len(skelsInfo)):
                	skeleton = SkeletonDetector.get_ith_skeleton(skelsInfo, ith_skel)
                	if ith_skel == 0:
                		my_detector.draw(show, humans)

            elif self.__flag_mode == 2:
                self.infoBox.setText(u'Multiplayer tracking')
                prediced_label = ''
                humans = my_detector.detect(show)
                skelsInfo = SkeletonDetector.humans_to_skelsInfo(humans)
                for ith_skel in range(0, len(skelsInfo)):
                	skeleton = SkeletonDetector.get_ith_skeleton(skelsInfo, ith_skel)
                	if ith_skel == 0:
                		my_detector.draw(show, humans)
                		print('show.shape: ', show.data)
                	drawActionResult(show, skeleton, prediced_label) #draw bboxs, prediced label
                #show = np.array(show)


            elif self.__flag_mode == 3:
                self.infoBox.setText(u'Sign language recognition')
                humans = my_detector.detect(show)
                skelsInfo_choose = SkeletonDetector.humans_to_skelsInfo_choose(humans, joint_choose)
                skelsInfo = SkeletonDetector.humans_to_skelsInfo(humans)
                if len(skelsInfo) == 0:
                    self.text = ''
                for ith_skel in range(0, len(skelsInfo)):
                    skeleton = SkeletonDetector.get_ith_skeleton(skelsInfo, ith_skel)
                    skeleton_choose = SkeletonDetector.get_ith_skeleton_choose(skelsInfo_choose, ith_skel)
                    print('skeleton_choose.shape: ', skeleton_choose)
                    if len(skeleton_choose) == 20:
                        prediced_label, acc = classifier.predict(skeleton_choose)
                        if acc < 0.95:
                            prediced_label = ''
                        if prediced_label != self.flag:
                            self.text = self.text + ' ' + prediced_label
                        self.flag = prediced_label
                    else:
                        prediced_label = ''
                    my_detector.draw(show, humans)
                    drawActionResult(show, skeleton, prediced_label)
                    self.textBox.setText('ID-1: ' + self.text)


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
    #action_classifier = load_action_premodel('Action/framewise_recognition.h5')
    print("Load all models done!")
    print("The system starts ro run.")
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
