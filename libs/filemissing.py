try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    print("Please Use PyQt5!")


class FileMissing(QDialog):

    def __init__(self, parent=None):
        super(FileMissing, self).__init__(parent)
        smallfont = QFont("Roboto", 8)
        self.setWindowTitle("File not Found!")
        self.button = QPushButton()
        pixmap = QPixmap("resources/icons/404.png")
        self.dummyicon = QLabel()
        self.dummyicon.setMaximumSize(30, 30)
        self.icon = QLabel()
        self.icon.setPixmap(pixmap)
        self.icon.setScaledContents(True)
        self.icon.setMaximumSize(60, 60)
        self.button.setText("Ok, Got it!")
        self.button.setMaximumSize(100,30)
        #self.button.setGeometry(QRect(300, 0, 100, 30))
        self.imperative = QLabel()
        self.imperative.setGeometry(QRect(100, 0, 350, 10))
        self.imperative.setText("Neither Annotation nor Weight File is Found!")
        self.imperative.setAlignment(Qt.AlignCenter)
        self.suggestion = QLabel()
        self.suggestion.setGeometry(QRect(0, 0, 250, 10))
        self.suggestion.setText("Suggestion: Annotate the Dataset First!")
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
        #self.button.setGeometry(QRect(100, 0, 350, 10))
        self.button.clicked.connect(self.action)
        self.setFixedSize(500,100)
        self.changetab = False

    def action(self):
        self.changetab = True
        self.reject()

    def isChangeTab(self):
        return self.changetab
