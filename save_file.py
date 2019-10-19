import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon

class SaveSummary(QWidget):

    def __init__(self, summary=''):
        super().__init__()
        self.title = 'Personalised Summarization Tool - Choose file location...'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.summary = summary
        self.saved = 0
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.saveFileDialog()
        
        self.close()
        
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save Summary","","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)
            
            with open(fileName, 'w') as f:
                f.write(self.summary)
                self.saved = 1
