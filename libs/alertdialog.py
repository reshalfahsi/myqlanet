try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    print("Please Use PyQt5!")


class AlertDialog(QDialog):

    def __init__(self, parent=None):
        super(AlertDialog, self).__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        smallfont = QFont("Roboto", 8)
        largefont = QFont("Roboto", 10)
        largefont.setBold(True)
        self.setWindowTitle("File not Found!")
        self.button = QPushButton()
        pixmap = QPixmap("resources/icons/404.png")
        self.dummyicon = QLabel()
        self.dummyicon.setMaximumSize(5, 5)
        self.icon = QLabel()
        self.icon.setPixmap(pixmap)
        self.icon.setScaledContents(True)
        self.icon.setMaximumSize(70, 70)
        self.button.setText("Ok, Got it!")
        self.button.setMaximumSize(100,25)
        self.imperative = QLabel()
        self.imperative.setGeometry(QRect(0, 0, 200, 10))
        self.imperative.setAlignment(Qt.AlignCenter)
        self.imperative.setFont(largefont)
        self.suggestion = QLabel()
        self.suggestion.setGeometry(QRect(0, 0, 200, 10))
        self.suggestion.setAlignment(Qt.AlignCenter)
        self.suggestion.setFont(smallfont)
        self.dummy = QPushButton()
        self.dummy.setVisible(False)
        layout = QHBoxLayout() 
        layout.addWidget(self.dummy)
        layout.addWidget(self.button)
        layout_ = QVBoxLayout()
        layout_.addWidget(self.imperative)
        layout_.addWidget(self.suggestion)
        _layout = QHBoxLayout()
        _layout.addWidget(self.dummyicon)
        _layout.addWidget(self.icon)
        _layout.addLayout(layout_)
        parent_layout = QVBoxLayout()
        parent_layout.addLayout(_layout)
        parent_layout.addLayout(layout)
        self.icon.setScaledContents(True)
        self.setLayout(parent_layout)
        self.button.clicked.connect(self.action)
        self.setFixedSize(300,100)

    def setStatus(self, status):
        if(status == 'predict_not_found'):
            self.imperative.setText("Weight File is not Valid!")
            self.suggestion.setText("Suggestion: Train the Model First!")
        elif(status == 'train_not_found'):
            self.imperative.setText("Annotation File is not Found!")
            self.suggestion.setText("Suggestion: Annotate the Model First!")
        elif(status == 'openfile_issue'):
            self.imperative.setText("File is Empty!")
            self.suggestion.setText("Suggestion: Open the File First!")
        elif(status == "saved"):
            self.setWindowTitle("File Saved")
            self.imperative.setText("Annotation File:")
            self.suggestion.setText("Successfully Saved!")
            pixmap = QPixmap("resources/icons/check.png")
            self.icon.setPixmap(pixmap)
            self.icon.setScaledContents(True)
        else:
            print('Status not Valid')

    def action(self):
        self.reject()
