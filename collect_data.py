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
import csv
count_frame = -30
poseEstimator = None

def load_model():
    global poseEstimator
    poseEstimator = TfPoseEstimator(
        get_graph_path('mobilenet_v2_large'), target_size=(432, 368)) # mobilenet_v2_large , mobilenet_v2_small , mobilenet_thin , cmu


class Ui_MainWindow(QtWidgets.QWidget):
    global count_frame
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
        self.setWindowTitle(u'Real-time multi-person attitude estimation and behavior recognition system') 

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
                self.button_mode_1.setText(u'Skeleton ON') 
                self.button_mode_2.setText(u'Tracking OFF')
                self.button_mode_3.setText(u'Recognition OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_1.setText(u'Skeleton OFF')
                self.infoBox.setText(u'Camera is on') 
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
                    msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check if the camera and computer are connected correctly.", 
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
        # if 'count_frame' is not locals():
        start = time.time()
        ret, frame = self.cap.read()
        show = cv2.resize(frame, (settings.winWidth, settings.winHeight))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        global count_frame
        if ret:
            if self.__flag_mode == 1:
                self.infoBox.setText(u'Current pose estimation model')
                humans = poseEstimator.inference(show)
                show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)  #ve khung xuong
                show, joints, bboxes, xcenter, sk, count_frame, joints_to_csv = TfPoseEstimator.get_skeleton(count_frame, joint_choose, show, humans, imgcopy=False) #lay thong tin khung xuong
                if count_frame > 0 and count_frame < 801 and len(joints_to_csv)==20:
                	cv2.putText(show, 'Bat dau ghi', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                	with open('train_10.csv', 'a') as f:
                		joints_to_csv.append(13)
                		writer = csv.writer(f)
                		writer.writerow(joints_to_csv)
                    	
                if count_frame > 800:
                	cv2.putText(show, 'Da luu du so sample', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elif self.__flag_mode == 2:
                self.infoBox.setText(u'Current multiplayer tracking mode') 
                humans = poseEstimator.inference(show)
                show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)
                show, joints, bboxes, xcenter, sk, count_frame = TfPoseEstimator.get_skeleton(count_frame, show, humans, imgcopy=False)
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
                self.infoBox.setText(u'Current human behavior recognition model') 
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
                    self.current = [i[-1] for i in trackers]

                    if len(self.previous) > 0:
                        for item in self.previous:
                            if item not in self.current and item in self.data:
                                del self.data[item]
                            if item not in self.current and item in self.memory:
                                del self.memory[item]

                    self.previous = self.current
                    for d in trackers:
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2])
                        ymax = int(d[3])
                        label = int(d[4])
                        try:
                            j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                        except:
                            j = 0
                        if joint_filter(joints[j]):
                            joints[j] = joint_completion(joint_completion(joints[j]))
                            if label not in self.data:
                                self.data[label] = [joints[j]]
                                self.memory[label] = 0
                            else:
                                self.data[label].append(joints[j])

                            if len(self.data[label]) == settings.L:
                                pred = actionPredictor().move_status(self.data[label])
                                if pred == 0:
                                    pred = self.memory[label]
                                else:
                                    self.memory[label] = pred
                                self.data[label].pop(0)

                                location = self.data[label][-1][1]
                                if location[0] <= 30:
                                    location = (51, location[1])
                                if location[1] <= 10:
                                    location = (location[0], 31)

                                cv2.putText(show, settings.move_status[pred], (location[0] - 30, location[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 255, 0), 2)

                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)

            end = time.time()
            self.fps = 1. / (end - start)
            cv2.putText(show, 'FPS: %.2f' % self.fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(show, 'frame: %.2f' % count_frame, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
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
	# with open('test.csv', 'a') as f:
	# 	writer = csv.writer(f)
	# 	csv_line = 'joints[0]_x, joints[0]_y, joints[1]_x, joints[1]_y, joints[2]_x, joints[2]_y, joints[3]_x, joints[3]_y, joints[4]_x, joints[4]_y, joints[5]_x, joints[5]_y, joints[6]_x, joints[6]_y, joints[7]_x, joints[7]_y, joints[8]_x, joints[8]_y, joints[11]_x, joints[11]_y, class'
	# 	writer.writerows([csv_line.split(',')])
	joint_choose = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
	load_model()
	print("Load all models done!")
	print("The system starts ro run.")
	app = QtWidgets.QApplication(sys.argv)
	ui = Ui_MainWindow()
	ui.show()
	sys.exit(app.exec_())
	f.close()
