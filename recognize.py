# -*- coding: UTF-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path
from utils.sort import Sort
from utils.actions import actionPredictor
from utils.joint_preprocess import *
#from Action.recognizer import load_action_premodel, framewise_recognize
import sys
import cv2
import numpy as np
import time
import settings
from keras.models import load_model
import mylib.io as myio
from mylib.display import drawActionResult
from mylib.data_preprocessing import pose_normalization
poseEstimator = None


def load_model():
    global poseEstimator, dnn_model
    poseEstimator = TfPoseEstimator(
        get_graph_path('mobilenet_v2_large'), target_size=(432, 368)) # mobilenet_v2_large , mobilenet_v2_small , mobilenet_thin , cmu
    dnn_model = load_model('model/action_recognition.h5')


########################################################################################################################################
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
    
def get_ith_skeleton(skelsInfo, ith_skeleton=0):
    return np.array(skelsInfo[ith_skeleton][1:1+18*2])

class ActionClassifier(object):
    
    def __init__(self):
        self.action_dict = ["kick", "punch", "squat", "stand", "wave"]
    def predict(self, skeleton):
        # Preprocess data
        tmp = pose_normalization(skeleton)
        skeleton_input = np.array(tmp).reshape(-1, len(tmp))
            
        # Predicted label: int & string
        predicted_idx = np.argmax(self.dnn_model.predict(skeleton_input))
        prediced_label = self.action_dict[predicted_idx]

        return prediced_label
############################################################################################################################################

class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.tracker = Sort(settings.sort_max_age, settings.sort_min_hit)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
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

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'Camera OFF') #相机

        self.button_mode_1 = QtWidgets.QPushButton(u'Skeleton OFF')  #姿态估计
        self.button_mode_2 = QtWidgets.QPushButton(u'Tracking OFF')
        self.button_mode_3 = QtWidgets.QPushButton(u'Recognition OFF')    #行为识别

        self.button_close = QtWidgets.QPushButton(u'Close') #退出

        self.button_open_camera.setMinimumHeight(50)
        self.button_mode_1.setMinimumHeight(50)
        self.button_mode_2.setMinimumHeight(50)
        self.button_mode_3.setMinimumHeight(50)

        self.button_close.setMinimumHeight(50)

        self.button_close.move(10, 100)

        self.infoBox = QtWidgets.QTextBrowser(self)
        self.infoBox.setGeometry(QtCore.QRect(10, 300, 200, 180))

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 200)

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
        self.setWindowTitle(u'Real-time multi-person attitude estimation and behavior recognition system') #实时多人姿态估计与行为识别系统

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_event)
        self.timer_camera.timeout.connect(self.show_camera)

        self.button_mode_1.clicked.connect(self.button_event)
        self.button_mode_2.clicked.connect(self.button_event)
        self.button_mode_3.clicked.connect(self.button_event)
        self.button_close.clicked.connect(self.close)

    def button_event(self):
        sender = self.sender()
        if sender == self.button_mode_1 and self.timer_camera.isActive():
            if self.__flag_mode != 1:
                self.__flag_mode = 1
                self.button_mode_1.setText(u'Skeleton ON') #姿态估计
                self.button_mode_2.setText(u'Tracking OFF')
                self.button_mode_3.setText(u'Recognition OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_1.setText(u'Skeleton OFF')
                self.infoBox.setText(u'Camera is on') #相机已打开
        elif sender == self.button_mode_2 and self.timer_camera.isActive():
            if self.__flag_mode != 2:
                self.__flag_mode = 2
                self.button_mode_1.setText(u'Skeleton OFF')
                self.button_mode_2.setText(u'Tracking ON')
                self.button_mode_3.setText(u'Recognition OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_2.setText(u'Tracking OFF')
                self.infoBox.setText(u'Camera is on')
        elif sender == self.button_mode_3 and self.timer_camera.isActive():
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
                self.infoBox.setText(u'Camera is off') #相机已关闭

    def show_camera(self):
        start = time.time()
        ret, frame = self.cap.read()
        show = cv2.resize(frame, (settings.winWidth, settings.winHeight))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        if ret:
            if self.__flag_mode == 1:
                self.infoBox.setText(u'Current pose estimation model') #当前为人体姿态估计模式
                humans = poseEstimator.inference(show)
                show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)
                a, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(show, humans, imgcopy=False) ###############
                print(joints)

            elif self.__flag_mode == 2:
                self.infoBox.setText(u'Current multiplayer tracking mode') #当前为多人跟踪模式
                humans = poseEstimator.inference(show)
                show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)
                show, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
                height = show.shape[0]
                width = show.shape[1]
                if bboxes:
                    result = np.array(bboxes)
                    det = result[:, 0:5]
                    det[:, 0] = det[:, 0] * width
                    det[:, 1] = det[:, 1] * height
                    det[:, 2] = det[:, 2] * width
                    det[:, 3] = det[:, 3] * height
                    trackers = self.tracker.update(det)

                    for d in trackers:
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2])
                        ymax = int(d[3])
                        label = int(d[4])
                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)

            elif self.__flag_mode == 3:
                self.infoBox.setText(u'Current human behavior recognition model') #当前为人体行为识别模式
                humans = poseEstimator.inference(show)
                ori = np.copy(show)
                show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)
                show, joints, bboxes, xcenter, sk= TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
                height = show.shape[0]
                width = show.shape[1]
                if bboxes:
                    result = np.array(bboxes)
                    det = result[:, 0:5]
                    det[:, 0] = det[:, 0] * width
                    det[:, 1] = det[:, 1] * height
                    det[:, 2] = det[:, 2] * width
                    det[:, 3] = det[:, 3] * height
                    trackers = self.tracker.update(det)
                    # self.current = [i[-1] for i in trackers]

                    # if len(self.previous) > 0:
                    #     for item in self.previous:
                    #         if item not in self.current and item in self.data:
                    #             del self.data[item]
                    #         if item not in self.current and item in self.memory:
                    #             del self.memory[item]

                    # self.previous = self.current
                    ###########################################################################################################################################################################
                    skelsInfo = humans_to_skelsInfo(humans, action_type = "unknown")
                    for ith_skel in range(0, len(skelsInfo)):
                        skeleton = get_ith_skeleton(skelsInfo, ith_skel)

                        # Classify action
                        classifier = ActionClassifier()
                        prediced_label = classifier.predict(skeleton)
                        print("prediced label is :", prediced_label)


                        # if 1:
                        #     # Draw skeleton
                        #     if ith_skel == 0:
                        #     #my_detector.draw(image_disp, humans)
                
                        #     # Draw bounding box and action type
                        #     drawActionResult(ori, skeleton, prediced_label)


                    ###########################################################################################################################################################################
                    for d in trackers:
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2])
                        ymax = int(d[3])
                        label = int(d[4])
                        # try:
                        #     j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                        # except:
                        #     j = 0
                        # if joint_filter(joints[j]):
                        #     joints[j] = joint_completion(joint_completion(joints[j]))
                        #     if label not in self.data:
                        #         self.data[label] = [joints[j]]
                        #         self.memory[label] = 0
                        #     else:
                        #         self.data[label].append(joints[j])

                        #     if len(self.data[label]) == settings.L:
                        #         pred = actionPredictor().move_status(self.data[label])
                        #         if pred == 0:
                        #             pred = self.memory[label]
                        #         else:
                        #             self.memory[label] = pred
                        #         self.data[label].pop(0)

                        #         location = self.data[label][-1][1]
                        #         if location[0] <= 30:
                        #             location = (51, location[1])
                        #         if location[1] <= 10:
                        #             location = (location[0], 31)

                        #         cv2.putText(show, settings.move_status[pred], (location[0] - 30, location[1] - 10),
                        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        #                     (0, 255, 0), 2)

                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)

            end = time.time()
            self.fps = 1. / (end - start)
            cv2.putText(show, 'FPS: %.2f' % self.fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"Shut down", u"Whether it is closed!") #关闭, 是否关闭！

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'OK') #确定
        cancel.setText(u'cancel') #取消
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
    load_model()
    #action_classifier = load_action_premodel('Action/framewise_recognition.h5')
    print("Load all models done!")
    print("The system starts ro run.")
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
