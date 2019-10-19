# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'webcam_start.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import time
import random
import subprocess

class progressThread(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(int)

    def __init__(self):
        QtCore.QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        while 1:
            closeWindow = 0
            with open("main_config", "r") as f:
                buffer = f.read().split()

            if buffer[3] == "stop":
                closeWindow = 1

            self.progress_update.emit(closeWindow)
            time.sleep(0.3)


class Ui_Dialog_webcam(object):
    def setupUi(self, Dialog):
        subprocess.Popen(["./opengazer"], stderr= subprocess.PIPE)
        Dialog.setObjectName("Dialog")
        Dialog.resize(463, 440)
        Dialog.setCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/Ink-Pen-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        Dialog.setStyleSheet("background-image:    url(:/images/shutterstock_1023246931_364607.jpg);\n"
"background-position: center center;\n"
"background-repeat:   no-repeat;")
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_2 = QtWidgets.QFrame(Dialog)
        self.frame_2.setStyleSheet("background-image: url(:/images/transparent-photoshop-background-grid-260nw-1023662581.jpg);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(111, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setStyleSheet("")
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(110, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.frame_2, 0, 0, 1, 1)
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setStyleSheet("background-image: url(:/images/transparent-photoshop-background-grid-260nw-1023662581.jpg);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.frame)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setTextVisible(False)
        self.progressBar.setOrientation(QtCore.Qt.Horizontal)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_3.addWidget(self.progressBar, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 1, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.dialog = Dialog

        self.progress_thread = progressThread()
        self.progress_thread.start()
        self.progress_thread.progress_update.connect(self.initialise)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Loading Webcam..."))
        self.label_2.setText(_translate("Dialog", "<html><head/><body><p><img src=\":/images/webcam.png\"/></p></body></html>"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; font-weight:600; color:#fce94f;\">Loading Webcam and Models...</span></p></body></html>"))

    def initialise(self, closeWindow):
        
        self.progressBar.setValue(random.randint(0,100))

        if closeWindow:
            QtWidgets.QDialog.close(self.dialog)

import main_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog_webcam()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

