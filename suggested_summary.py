# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'suggested_summary.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog_suggSummary(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(556, 45)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/Ink-Pen-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        Dialog.setStyleSheet("background-color: rgb(114, 159, 207);")
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.suggSummary = QtWidgets.QLineEdit(Dialog)
        self.suggSummary.setStyleSheet("background-color: rgb(238, 238, 236);")
        self.suggSummary.setClearButtonEnabled(False)
        self.suggSummary.setObjectName("suggSummary")
        self.gridLayout.addWidget(self.suggSummary, 0, 0, 1, 1)
        self.genSummary = QtWidgets.QPushButton(Dialog)
        self.genSummary.setObjectName("genSummary")
        self.gridLayout.addWidget(self.genSummary, 0, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Suggested Summary"))
        self.suggSummary.setPlaceholderText(_translate("Dialog", "Article Title"))
        self.genSummary.setText(_translate("Dialog", "Generate Summary"))

import main_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog_suggSummary()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
