try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    print("Please Use PyQt5!")

class Canvas(QLabel):
    def __init__(self, parent):
        super(Canvas, self).__init__(parent)
        self.begin = QPoint()
        self.end = QPoint()
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.active = False
    
    def getRect(self):
        return (self.x, self.y, self. width, self.height)

    def deleteRect(self):
        self.begin = QPoint(0,0)
        self.end = QPoint(0,0)
        self.update()

    def setActive(self, status):
        self.active = status

    def paintEvent(self, event):
        if(self.active):
            qp = QPainter(self)
            qp.setPen(QPen(QColor(0, 255, 0, 100), 3, Qt.SolidLine))
            br = QBrush(QColor(100, 255, 100, 40))  
            qp.setBrush(br)
            rect = QRect(self.begin, self.end)
            self.x = rect.x()
            self.y = rect.y()
            self.width = rect.width()
            self.height = rect.height()   
            qp.drawRect(rect)
            qp.end()       

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.update()
