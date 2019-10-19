# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'time_elapsed2.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from Frame_recording import Screen
from threading import Thread
import settings
import subprocess
import warnings
import os
import signal

warnings.filterwarnings('ignore')

TICK_TIME = 2**6

rThread = 0
eyeGaze = 0

class recordThread(QtCore.QThread):

    def __init__(self):
        QtCore.QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        rThread = subprocess.Popen(["python3", "../text_extraction/Frame_recording.py"])

class gazeThread(QtCore.QThread):

    def __init__(self):
        QtCore.QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        
        eyeGaze = subprocess.Popen(["./opengazer"], stdout= subprocess.PIPE)
        with open("gaze_PID", "w") as f:
            f.write(str(eyeGaze.pid))

        gaze_points = eyeGaze.communicate()[0]
        with open("gaze_points.csv", "w") as f:
            f.write(gaze_points.decode('utf-8'))
    
class Ui_Dialog_time_elapsed(Qt.QMainWindow, object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(248, 151)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/I5nk-Pen-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        Dialog.setStyleSheet("background-image: url(:/images/shutterstock_1023246931_364607.jpg);\n"
"background-position: center center;\n"
"background-repeat:   no-repeat;")
        self.gridLayout_3 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setStyleSheet("background-image: url(:/images/transparent-photoshop-background-grid-260nw-1023662581.jpg);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.toolButton = QtWidgets.QToolButton(self.frame)
        self.toolButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/images.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton.setIcon(icon1)
        self.toolButton.setObjectName("toolButton")
        self.gridLayout.addWidget(self.toolButton, 0, 1, 1, 1)
        self.lcdNumber = QtWidgets.QLCDNumber(self.frame)
        font = QtGui.QFont()
        font.setKerning(True)
        self.lcdNumber.setFont(font)
        self.lcdNumber.setCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.lcdNumber.setStyleSheet("color: red;")
        self.lcdNumber.setFrameShape(QtWidgets.QFrame.Box)
        self.lcdNumber.setSmallDecimalPoint(False)
        self.lcdNumber.setDigitCount(10)
        self.lcdNumber.setObjectName("lcdNumber")
        self.gridLayout.addWidget(self.lcdNumber, 1, 0, 1, 2)
        self.gridLayout_3.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        # self.sc = Screen()

        self.toolButton.clicked.connect(self.recording)

        self.record_Thread = recordThread()

        self.gaze_thread = gazeThread()

        self.dialog = Dialog
        self.timer = Qt.QTimer()
        self.timer.setInterval(TICK_TIME)
        self.timer.timeout.connect(self.tick)

        self.notRecording()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Time Elapsed"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Not Recording</span></p></body></html>"))
        self.toolButton.setText(_translate("Dialog", "Start Recording"))

    def display(self):
        self.lcdNumber.display("%d:%05.2f"%(self.time//60, self.time%60))

    @Qt.pyqtSlot()
    def tick(self):
        self.time += TICK_TIME/1000
        self.display()

    def recording(self):
        with open("main_config", "r") as f:
            buffer = f.read().split()
            
        with open("main_config", "w") as f:
            buffer[0] = "yes"
            buffer[1] = "yes"
            f.write("\n".join(buffer))
        
        
        self.gaze_thread.start()
        self.record_Thread.start()

        self.timer.start()
        _translate = QtCore.QCoreApplication.translate
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/images_stop.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton.setIcon(icon1)
        self.toolButton.clicked.disconnect()
        self.label.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; color:#ff0000;\">Recording......</span></p></body></html>"))
        self.toolButton.clicked.connect(self.stopedRecording)
        # self.threadTimer = Thread(target=self.sc.start_recording, args=())
        # self.threadTimer.start()
        # threadRecorder = Thread(target= self.record, args=())
        # threadRecorder.start()
        # self.sc.start_recording()

        # self.record()
    
    
    # def record(self):
    #     while True:
    #         if self.sc.new_frame:
    #             print('Recording!')
    #             self.sc.capture_Frames()
    #             # time.sleep(0.1)


    @Qt.pyqtSlot()
    def stopedRecording(self):
        settings.stop_recording()
        self.timer.stop()

        # global rThread
        # os.killpg(os.getpgid(rThread.pid), signal.SIGTERM)

        # global gazeThread
        # os.killpg(os.getpgid(gazeThread.pid), signal.SIGTERM)
        with open("main_config", "r") as f:
            buffer = f.read().split()
            
        with open("main_config", "w") as f:
            buffer[0] = "no"
            buffer[1] = "no"
            f.write("\n".join(buffer))


        # self.threadTimer.join()
        # self.sc.on_stop_recording()
        QtWidgets.QDialog.accept(self.dialog)
        #QtWidgets.QDialog.close(self.dialog)

    @Qt.pyqtSlot()
    def notRecording(self):
        self.time = 0
        self.display()

    # def getSummary(self):
    #     return self.sc.summary


import time_elapsed_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog_time_elapsed()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
    
