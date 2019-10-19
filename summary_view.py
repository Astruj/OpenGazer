# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'summary_view.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from exit_confirm import Ui_Dialog_exit_confirm
from save_file import SaveSummary

class Ui_Dialog_summary_view(object):
    def setupUi(self, Dialog, summary=""):
        Dialog.setObjectName("Dialog")
        Dialog.resize(783, 461)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/Ink-Pen-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        Dialog.setStyleSheet("background-image:    url(:/images/shutterstock_1023246931_364607.jpg);\n"
"background-position: center center;\n"
"background-repeat:   no-repeat;")
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.summaryDisplay = QtWidgets.QTextBrowser(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.summaryDisplay.setFont(font)
        self.summaryDisplay.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.summaryDisplay.setAcceptDrops(False)
        self.summaryDisplay.setStyleSheet("background-image: url(:/images/transparent-photoshop-background-grid-260nw-1023662581.jpg);")
        self.summaryDisplay.setFrameShape(QtWidgets.QFrame.Box)
        self.summaryDisplay.setFrameShadow(QtWidgets.QFrame.Plain)
        self.summaryDisplay.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.summaryDisplay.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.summaryDisplay.setDocumentTitle("")
        self.summaryDisplay.setUndoRedoEnabled(True)
        self.summaryDisplay.setReadOnly(False)
        self.summaryDisplay.setCursorWidth(1)
        self.summaryDisplay.setOpenExternalLinks(True)
        self.summaryDisplay.setObjectName("summaryDisplay")
        self.gridLayout_2.addWidget(self.summaryDisplay, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_2, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.frame)
        self.buttonBox.setStyleSheet("background-image: url(:/images/transparent-photoshop-background-grid-260nw-1023662581.jpg);")
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_3.addWidget(self.buttonBox, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        with open("./File_out/summary.txt", "r") as f:
            summary = f.read()
            self.summaryDisplay.setText(summary)

        self.saved = 0
        self.dialog = Dialog
        self.buttonBox.accepted.connect(self.save_it)
        self.buttonBox.rejected.connect(self.exit_it)

    def save_it(self):
        s = SaveSummary(self.summaryDisplay.toPlainText())
        self.saved = 1
        
        if s.saved:
            QtWidgets.QDialog.close(self.dialog)

    def exit_it(self):

        rsp = 0

        if self.saved:
            QtWidgets.QDialog.close(self.dialog)
        else:
            Dialog_exit = QtWidgets.QDialog()
            ui_exit = Ui_Dialog_exit_confirm()
            ui_exit.setupUi(Dialog_exit)
            Dialog_exit.show()
            rsp = Dialog_exit.exec_()

        if rsp == QtWidgets.QDialog.Accepted:
            QtWidgets.QDialog.close(self.dialog)
        
        else: 
            pass


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Summary"))
        self.summaryDisplay.setPlaceholderText(_translate("Dialog", "Your summary is displayed here......."))


import summary_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog_summary_view()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
