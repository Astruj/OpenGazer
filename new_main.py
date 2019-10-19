# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'new_main.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from time_elapsed import Ui_Dialog_time_elapsed
from summarising import Ui_Dialog_summarising
from summary_view import Ui_Dialog_summary_view
from random import choice
from save_file import SaveSummary
from exit_confirm import Ui_Dialog_exit_confirm
from running import Ui_Dialog_running
from wait import Ui_Dialog_wait
from suggested_summary import Ui_Dialog_suggSummary
from webcam_start import Ui_Dialog_webcam
import subprocess
import sys
import warnings
from createSummary import crMain
from gazePlotMap import gPMain
import psutil

warnings.filterwarnings('ignore')

quotes = ['"Egotism is the source of all faults and miseries" - Thomas Carlyle'] #, '"The saddest summary of a life contains three descriptions: could have, might have, and should have." - Louis E. Boone']


class shellThread(QtCore.QThread):

    def __init__(self):
        QtCore.QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        gPMain()
        crMain()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1658, 864)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/Ink-Pen-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setStyleSheet("background: url(:/images/shutterstock_1023246931_364607.jpg)")
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(270, 320, 382, 74))
        self.frame_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.quotes = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(8)
        self.quotes.setFont(font)
        self.quotes.setAlignment(QtCore.Qt.AlignCenter)
        self.quotes.setObjectName("quotes")
        self.gridLayout_3.addWidget(self.quotes, 2, 0, 1, 1)
        self.analyseCommandLink = QtWidgets.QCommandLinkButton(self.frame_3)
        self.analyseCommandLink.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.analyseCommandLink.setCheckable(False)
        self.analyseCommandLink.setObjectName("analyseCommandLink")
        self.gridLayout_3.addWidget(self.analyseCommandLink, 1, 0, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(130, 70, 741, 181))
        self.frame_2.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.frame_2.setMouseTracking(False)
        self.frame_2.setAutoFillBackground(False)
        self.frame_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_2.setObjectName("frame_2")
        self.title = QtWidgets.QLabel(self.frame_2)
        self.title.setEnabled(True)
        self.title.setGeometry(QtCore.QRect(170, 20, 531, 44))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(26)
        self.title.setFont(font)
        self.title.setObjectName("title")
        self.summariseButton = QtWidgets.QPushButton(self.frame_2)
        self.summariseButton.setGeometry(QtCore.QRect(140, 86, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Oswald,Helvetica")
        font.setPointSize(18)
        font.setUnderline(False)
        font.setStrikeOut(False)
        self.summariseButton.setFont(font)
        self.summariseButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.summariseButton.setAcceptDrops(False)
        self.summariseButton.setStyleSheet("position: relative;\n"
"    width:220px;\n"
"    height:40px;\n"
"    text-align:center;\n"
"    color:#FFF;\n"
"    text-decoration:none;\n"
"    line-height:43px;\n"
"    font-family:\'Oswald\', Helvetica;\n"
"    margin: 30px;\n"
"    background: #3EACBA;\n"
"    border:1px solid #379AA4;\n"
"    background-image:-webkit-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:-moz-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:-ms-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:-o-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:linear-gradient(top, #48C6D4, #3EACBA);\n"
"    \n"
"    border-radius:5px;\n"
"")
        self.summariseButton.setObjectName("summariseButton")
        self.lengthCombo = QtWidgets.QComboBox(self.frame_2)
        self.lengthCombo.setGeometry(QtCore.QRect(140, 130, 321, 23))
        font = QtGui.QFont()
        font.setFamily("Oswald,Helvetica")
        font.setPointSize(12)
        font.setUnderline(False)
        font.setStrikeOut(False)
        self.lengthCombo.setFont(font)
        self.lengthCombo.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.lengthCombo.setStyleSheet("position: relative;\n"
"    width:220px;\n"
"    height:40px;\n"
"    text-align:center;\n"
"    color:#000;\n"
"    text-decoration:none;\n"
"    line-height:43px;\n"
"    font-family:\'Oswald\', Helvetica;\n"
"    margin: 30px;\n"
"    background: #3EACBA;\n"
"    border:1px solid #379AA4;\n"
"    background-image:-webkit-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:-moz-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:-ms-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:-o-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:linear-gradient(top, #48C6D4, #3EACBA);\n"
"    border-radius:5px;\n"
)
        self.lengthCombo.setObjectName("lengthCombo")
        self.lengthCombo.addItem("")
        self.lengthCombo.addItem("")
        self.lengthCombo.addItem("")
        self.lengthCombo.addItem("")
        self.frame_4 = QtWidgets.QFrame(self.frame_2)
        self.frame_4.setGeometry(QtCore.QRect(20, 20, 131, 131))
        self.frame_4.setStyleSheet("background-image: url(:/images/Ink-Pen-icon.png);")
        self.frame_4.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.analyseCommandLink.setStyleSheet("position: relative;\n"
"    width:220px;\n"
"    height:40px;\n"
"    text-align:center;\n"
"    color:#000;\n"
"    text-decoration:none;\n"
"    line-height:43px;\n"
"    font-family:\'Oswald\', Helvetica;\n"
"    margin: 30px;\n"
"    background: #3EACBA;\n"
"    border:1px solid #379AA4;\n"
"    background-image:-webkit-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:-moz-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:-ms-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:-o-linear-gradient(top, #48C6D4, #3EACBA);\n"
"    background-image:linear-gradient(top, #48C6D4, #3EACBA);\n"
"    border-radius:5px;\n"
)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.isRunning = 0
        self.quotes.setText(choice(quotes))
        self.shell_thread = shellThread()
        self.summariseButton.clicked.connect(self.summarise)

        self.analyseCommandLink.clicked.connect(self.calibrate)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Personalised Summary Tool"))
        self.quotes.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-style:italic;\">&quot;Egotism is the source of all faults and miseries&quot; - Thomas Carlyle</span></p></body></html>"))
        self.analyseCommandLink.setText(_translate("MainWindow", "    Suggested Summary"))
        self.title.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:28pt;\">Personalised Summary Tool</span></p></body></html>"))
        self.summariseButton.setText(_translate("MainWindow", "Summarise"))
        self.lengthCombo.setItemText(0, _translate("MainWindow", "Choose Summary length..."))
        self.lengthCombo.setItemText(1, _translate("MainWindow", "Short Summary"))
        self.lengthCombo.setItemText(2, _translate("MainWindow", "Medium Summary"))
        self.lengthCombo.setItemText(3, _translate("MainWindow", "Long Summary"))


    def summarise(self):

        with open("main_config", "r") as f:
            buff = f.read().split()

        if buff[2] == "yes":

            if self.isRunning:
                Dialog = QtWidgets.QDialog()
                ui = Ui_Dialog_running()
                ui.setupUi(Dialog)
                Dialog.show()
                Dialog.exec_()
                del Dialog
            
            else:
                self.isRunning = 1

                length = self.lengthCombo.currentIndex()

                #print(f"Length:{length}")

                # Dialog = QtWidgets.QDialog()
                # ui = Ui_Dialog_webcam()
                # ui.setupUi(Dialog)
                # Dialog.show()
                # Dialog.exec_()

                #Runs the time Elapsed Window
                Dialog = QtWidgets.QDialog()
                ui = Ui_Dialog_time_elapsed()
                ui.setupUi(Dialog)
                # summary = ui.getSummary()
                summary = "lol"
                Dialog.show()
                rspt = Dialog.exec_()
                
                print(rspt)

                if rspt == QtWidgets.QDialog.Rejected:
                    self.isRunning = 0

                else:

                    with open("gaze_PID", "r") as f:
                        gazePID = int(f.read())

                    while 1:
                        if not psutil.pid_exists(gazePID):
                            break

                    self.shell_thread.start()

                    #Opens Summarising window
                    Dialog = QtWidgets.QDialog()
                    ui = Ui_Dialog_summarising()
                    ui.setupUi(Dialog, length, summary)
                    Dialog.show()
                    Dialog.exec_()
                    del Dialog

                    # self.shell_thread.terminate()

                    with open("summary_created.txt", "r") as f:
                        check = f.read()

                    if check == '1':
                        print("Reading summary")

                        #Opens summary view
                        Dialog = QtWidgets.QDialog()
                        ui = Ui_Dialog_summary_view()
                        ui.setupUi(Dialog, summary)
                        Dialog.show()
                        rsp = Dialog.exec_()
                        config = open("./summary_created.txt", "w")
                        config.write("0")
                        config.close()
                        if rsp == QtWidgets.QDialog.Accepted:
                            self.isRunning = 0
                            print(f"from main {rsp}")

                        else:
                            self.isRunning = 0
                            # print(f"from main{rsp}")
                            # Dialog_exit = QtWidgets.QDialog()
                            # ui_exit = Ui_Dialog_exit_confirm()
                            # ui_exit.setupUi(Dialog_exit)
                            # Dialog_exit.show()
                            # Dialog_exit.exec_()
                        del Dialog
                    else:
                        print("Summary not created")
                
                # #Opens Summarising window
                # Dialog = QtWidgets.QDialog()
                # ui = Ui_Dialog_summarising()
                # ui.setupUi(Dialog, length, summary)
                # Dialog.show()
                # Dialog.exec_()
                # del Dialog

                # #Opens summary view
                # Dialog = QtWidgets.QDialog()
                # ui = Ui_Dialog_summary_view()
                # ui.setupUi(Dialog, summary)
                # Dialog.show()
                # rsp = Dialog.exec_()

                # if rsp == QtWidgets.QDialog.Accepted:
                #     self.isRunning = 0
                #     print(f"from main {rsp}")

                # else:
                #     self.isRunning = 0
                #     # print(f"from main{rsp}")
                #     # Dialog_exit = QtWidgets.QDialog()
                #     # ui_exit = Ui_Dialog_exit_confirm()
                #     # ui_exit.setupUi(Dialog_exit)
                #     # Dialog_exit.show()
                #     # Dialog_exit.exec_()        

        # else:
        #     Dialog = QtWidgets.QDialog()
        #     ui = Ui_Dialog_pcalb()
        #     ui.setupUi(Dialog)
        #     Dialog.show()
        #     Dialog.exec_()
        #     del Dialog

    def calibrate(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog_suggSummary()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

import main_rc

def appExit():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    app.exec_()
    with open("main_config", "r") as f:
        buffer = f.read().split()
                
    with open("main_config", "w") as f:
        buffer[0] = "stop"
        f.write("\n".join(buffer))

if __name__ == "__main__":
    
    with open("main_config", "r") as f:
        buffer = f.read().split()
                
    with open("main_config", "w") as f:
        buffer[0] = "o"
        buffer[3] = "o"
        f.write("\n".join(buffer))
        

    #subprocess.Popen(["python3", "capture_gaze.py"], stderr= subprocess.PIPE)
    # app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    # ui = Ui_MainWindow()
    # ui.setupUi(MainWindow)
    # MainWindow.show()
    # sys.exit(app.exec_())
    sys.exit(appExit())