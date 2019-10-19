# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'summarising.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import time

class progressThread(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(int)

    def __init__(self):
        QtCore.QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        while 1:
            maxVal = 1
            self.progress_update.emit(maxVal)
            time.sleep(0.27)



class Ui_Dialog_summarising(object):
    def setupUi(self, Dialog, length="", summary=""):
        Dialog.setObjectName("Dialog")
        Dialog.resize(563, 237)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/Ink-Pen-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        Dialog.setStyleSheet("background-image:    url(:/images/shutterstock_1023246931_364607.jpg);\n"
"background-position: center center;\n"
"background-repeat:   no-repeat;")
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setStyleSheet("background-image: url(:/images/transparent-photoshop-background-grid-260nw-1023662581.jpg);")
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.frame)
        self.progressBar.setEnabled(True)
        self.progressBar.setProperty("value", 90)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_2.addWidget(self.progressBar, 1, 0, 1, 1)
        self.progressListView = QtWidgets.QListView(self.frame)
        self.progressListView.setStyleSheet("background-image: url(:/images/transparent-photoshop-background-grid-260nw-1023662581.jpg);")
        self.progressListView.setObjectName("progressListView")
        self.gridLayout_2.addWidget(self.progressListView, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.dialog = Dialog     
        self.summaryLength = length
        print(f"Length:{length}")

        self.progressBar.setValue(0)

        self.progress_thread = progressThread()
        self.progress_thread.start()
        self.progress_thread.progress_update.connect(self.processing)


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Summarising"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt;\">Summarising....</span></p></body></html>"))

    def processing(self, maxVal):
        model = QtGui.QStandardItemModel()
        self.progressListView.setModel(model)
        item = QtGui.QStandardItem("Extracting text...")
        item1 = QtGui.QStandardItem("Removing Redundancy...")
        item2 = QtGui.QStandardItem("Cleaning Text...")
        item3 = QtGui.QStandardItem("Performing Extractive Summarisation...")
        # item4 = QtGui.QStandardItem("Doing Stuff")

        # model.appendRow(item)
        # self.progressBar.setValue(20)
        # model.appendRow(item1)
        # self.progressBar.setValue(40)
        # model.appendRow(item2)
        # self.progressBar.setValue(60)
        # model.appendRow(item3)
        # self.progressBar.setValue(80)
        # # model.appendRow(item4)
        # # self.progressBar.setValue(100)

        self.progressBar.setValue(self.progressBar.value() + maxVal)

        with open("./summary_created.txt", "r") as f:
            check = f.read()

        if self.progressBar.value() <25:
            model.appendRow(item)
        elif self.progressBar.value() < 50:
            model.appendRow(item)
            model.appendRow(item1)
        elif self.progressBar.value() < 75:
            model.appendRow(item)
            model.appendRow(item1)
            model.appendRow(item2)
        elif self.progressBar.value() < 99:
            model.appendRow(item)
            model.appendRow(item1)
            model.appendRow(item2)
            model.appendRow(item3)
        elif self.progressBar.value() == 99:
            if check == "1":
                QtWidgets.QDialog.close(self.dialog)
            else:
                self.progressBar.setValue(97)

        if maxVal == 0:
            self.progressBar.setValue(100)


import summarising_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog_summarising()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
