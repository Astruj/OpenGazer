# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calibration2.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from cal_success import Ui_Dialog_success
#buttonNames = ["00", "01", "02", "12", "22", "21", "20", "10", "11"]

class Ui_Dialog_calibration(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(853, 480)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/Ink-Pen-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        #Dialog.setStyleSheet("background-image: url(:/images/shutterstock_1023246931_364607.jpg);\n")
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setStyleSheet("background-image: url(:/images/transparent-photoshop-background-grid-260nw-1023662581.jpg);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.button_00 = QtWidgets.QToolButton(self.frame)
        self.button_00.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.button_00.setStyleSheet("background-color: #FF3535;\n"
"border: none;\n"
"color: white;\n"
"padding: 2px;\n"
"text-align: center;\n"
"text-decoration: none;\n"
"font-size: 16px;\n"
"margin: 0px 0px;\n"
"border-radius: 8%;\n"
"height: 10px;\n"
"width: 10px;")
        self.button_00.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.button_00.setObjectName("button_00")
        self.gridLayout_2.addWidget(self.button_00, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(358, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 0, 1, 1, 1)
        self.button_01 = QtWidgets.QToolButton(self.frame)
        self.button_01.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.button_01.setStyleSheet("background-color: #FF3535;\n"
"border: none;\n"
"color: white;\n"
"padding: 2px;\n"
"text-align: center;\n"
"text-decoration: none;\n"
"font-size: 16px;\n"
"margin: 0px 0px;\n"
"border-radius: 8%;\n"
"height: 10px;\n"
"width: 10px;")
        self.button_01.setObjectName("button_01")
        self.gridLayout_2.addWidget(self.button_01, 0, 2, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(358, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 0, 3, 1, 1)
        self.button_02 = QtWidgets.QToolButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(0)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.button_02.setFont(font)
        self.button_02.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.button_02.setStyleSheet("background-color: #FF3535;\n"
"border: none;\n"
"color: white;\n"
"padding: 2px;\n"
"text-align: center;\n"
"text-decoration: none;\n"
"font-size: 16px;\n"
"margin: 0px 0px;\n"
"border-radius: 8%;\n"
"height: 10px;\n"
"width: 10px;")
        self.button_02.setObjectName("button_02")
        self.gridLayout_2.addWidget(self.button_02, 0, 4, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 173, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem2, 1, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 173, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem3, 1, 2, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 173, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem4, 1, 4, 1, 1)
        self.button_10 = QtWidgets.QToolButton(self.frame)
        self.button_10.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.button_10.setStyleSheet("background-color: #FF3535;\n"
"border: none;\n"
"color: white;\n"
"padding: 2px;\n"
"text-align: center;\n"
"text-decoration: none;\n"
"font-size: 16px;\n"
"margin: 0px 0px;\n"
"border-radius: 8%;\n"
"height: 10px;\n"
"width: 10px;")
        self.button_10.setObjectName("button_10")
        self.gridLayout_2.addWidget(self.button_10, 2, 0, 1, 1)
        self.button_11 = QtWidgets.QToolButton(self.frame)
        self.button_11.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.button_11.setStyleSheet("background-color: #FF3535;\n"
"border: none;\n"
"color: white;\n"
"padding: 2px;\n"
"text-align: center;\n"
"text-decoration: none;\n"
"font-size: 16px;\n"
"margin: 0px 0px;\n"
"border-radius: 8%;\n"
"height: 10px;\n"
"width: 10px;")
        self.button_11.setObjectName("button_11")
        self.gridLayout_2.addWidget(self.button_11, 2, 2, 1, 1)
        self.button_12 = QtWidgets.QToolButton(self.frame)
        self.button_12.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.button_12.setStyleSheet("background-color: #FF3535;\n"
"border: none;\n"
"color: white;\n"
"padding: 2px;\n"
"text-align: center;\n"
"text-decoration: none;\n"
"font-size: 16px;\n"
"margin: 0px 0px;\n"
"border-radius: 8%;\n"
"height: 10px;\n"
"width: 10px;")
        self.button_12.setObjectName("button_12")
        self.gridLayout_2.addWidget(self.button_12, 2, 4, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 173, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem5, 3, 0, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 173, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem6, 3, 2, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(20, 173, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem7, 3, 4, 1, 1)
        self.button_20 = QtWidgets.QToolButton(self.frame)
        self.button_20.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.button_20.setStyleSheet("background-color: #FF3535;\n"
"border: none;\n"
"color: white;\n"
"padding: 2px;\n"
"text-align: center;\n"
"text-decoration: none;\n"
"font-size: 16px;\n"
"margin: 0px 0px;\n"
"border-radius: 8%;\n"
"height: 10px;\n"
"width: 10px;")
        self.button_20.setObjectName("button_20")
        self.gridLayout_2.addWidget(self.button_20, 4, 0, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(358, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem8, 4, 1, 1, 1)
        self.button_21 = QtWidgets.QToolButton(self.frame)
        self.button_21.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.button_21.setStyleSheet("background-color: #FF3535;\n"
"border: none;\n"
"color: white;\n"
"padding: 2px;\n"
"text-align: center;\n"
"text-decoration: none;\n"
"font-size: 16px;\n"
"margin: 0px 0px;\n"
"border-radius: 8%;\n"
"height: 10px;\n"
"width: 10px;")
        self.button_21.setObjectName("button_21")
        self.gridLayout_2.addWidget(self.button_21, 4, 2, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(358, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem9, 4, 3, 1, 1)
        self.button_22 = QtWidgets.QToolButton(self.frame)
        self.button_22.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.button_22.setStyleSheet("background-color: #FF3535;\n"
"border: none;\n"
"color: white;\n"
"padding: 2px;\n"
"text-align: center;\n"
"text-decoration: none;\n"
"font-size: 16px;\n"
"margin: 0px 0px;\n"
"border-radius: 8%;\n"
"height: 10px;\n"
"width: 10px;")
        self.button_22.setObjectName("button_22")
        self.gridLayout_2.addWidget(self.button_22, 4, 4, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem10, 2, 1, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem11, 2, 3, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)


        self.buttons = [self.button_00, self.button_01, self.button_02, self.button_12, self.button_22, self.button_21, self.button_20, self.button_10, self.button_11]

        self.button_01.hide()
        self.button_02.hide()
        self.button_10.hide()
        self.button_11.hide()
        self.button_12.hide()
        self.button_20.hide()
        self.button_21.hide()
        self.button_22.hide()

        #self.positions = {} #stores the pixel coordinates of all the buttons

        self.dialog = Dialog
        QtWidgets.QDialog.showMaximized(self.dialog)

        self.buttonIndex = 0
        self.buttons = [self.button_00, self.button_01, self.button_02, self.button_12, self.button_22, self.button_21, self.button_20, self.button_10, self.button_11]
        self.clicked = 4

        self.button_00.clicked.connect(self.clicking)


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Calibration"))
        self.button_00.setText(_translate("Dialog", "5"))
        self.button_01.setText(_translate("Dialog", "5"))
        self.button_02.setText(_translate("Dialog", "5"))
        self.button_10.setText(_translate("Dialog", "5"))
        self.button_11.setText(_translate("Dialog", "5"))
        self.button_12.setText(_translate("Dialog", "5"))
        self.button_20.setText(_translate("Dialog", "5"))
        self.button_21.setText(_translate("Dialog", "5"))
        self.button_22.setText(_translate("Dialog", "5"))

    #Things to do while clicking
    def clicking(self):
        
        if self.clicked == 0:
            self.buttons[self.buttonIndex].clicked.disconnect()
            self.buttons[self.buttonIndex].setText(str(self.clicked))
            
            x = (self.frame.x() + self.buttons[self.buttonIndex].x() + self.buttons[self.buttonIndex].width()//2, self.frame.y() + self.buttons[self.buttonIndex].y() + self.buttons[self.buttonIndex].height()//2)
            


            with open("config", "r") as f:
                buffer = f.read().split()
            
            with open("config", "w") as f:
                buffer[3] = "yes"
                ok = "\n".join(buffer)
                print(f'{ok} -> calibration w')
                f.write("\n".join(buffer))
            
            print(x)
            with open("gaze.txt", "a") as f:
                f.write(str(x) + "\n")
            
            if self.buttonIndex < 8:
                self.buttonIndex += 1
                self.buttons[self.buttonIndex].show()
                self.clicked = 4
                self.buttons[self.buttonIndex].clicked.connect(self.clicking)
            else:
                with open("config", "w") as f:
                    f.write("stop")
                self.calibrate() #When all the buttons are clicked
            
        else:

            with open("config", "w") as f:
                f.write("ok\nok\nok\nno\nok")
            
            if int(self.buttons[self.buttonIndex].text()) < 5:
                with open("config", "r") as f:
                    buffer = f.read().split()
                    print(f'{buffer} -> calibration r')

                with open("config", "w") as f:
                    buffer[4] = "yes"
                    f.write("\n".join(buffer))


            self.buttons[self.buttonIndex].setText(str(self.clicked))
            self.clicked -= 1
    
    def calibrate(self):
        # for button in self.buttons:
        #     self.positions[button] = button.pos

        #stores the pixel values here:
        # for i in range(0,9):
        #     self.positions[buttonNames[i]] = (self.frame.x() + self.buttons[i].x() + self.buttons[i].width()//2, self.frame.y() + self.buttons[i].y() + self.buttons[i].height()//2)
        
        with open("main_config", "r") as f:
            buffer = f.read().split()
            
        with open("main_config", "w") as f:
            buffer[2] = "yes"
            f.write("\n".join(buffer))


        with open("config", "w") as f:
            f.write("stop\nok\nok\nok\nok")
        
        QtWidgets.QDialog.close(self.dialog)#Closes the window

        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog_success()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()
        del Dialog
        

        #print(self.positions) #Prints the dictionary


import main_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog_calibration()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
