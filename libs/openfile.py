try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    print("Please Use PyQt5!")

class OpenFile(QDialog):

    def __init__(self, parent=None):
        super(OpenFile, self).__init__(parent)
        self.setWindowTitle("Open File..")
        self.button_folder = QPushButton("Folder")
        self.button_file = QPushButton("File")
        self.imperative = QLabel()
        self.imperative.setGeometry(QRect(0, 0, 250, 50))
        self.imperative.setText("Please Select the Source Path: ")
        self.imperative.setAlignment(Qt.AlignCenter)
        layout = QHBoxLayout()
        layout.addWidget(self.button_folder)
        layout.addWidget(self.button_file)
        parent_layout = QVBoxLayout()
        parent_layout.addWidget(self.imperative)
        parent_layout.addLayout(layout)
        self.setLayout(parent_layout)
        self.button_folder.clicked.connect(self.folder)
        self.button_file.clicked.connect(self.file)
        self.filenames = ''
        self.setFixedSize(250,120)

    def file(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
		
        if dlg.exec_():
            self.filenames = dlg.selectedFiles()
            self.filenames = self.filenames[0]
            self.reject()

    def folder(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
		
        if dlg.exec_():
            self.filenames = dlg.selectedFiles()
            self.filenames = self.filenames[0]
            self.reject()

    def getFileName(self):
        return self.filenames
