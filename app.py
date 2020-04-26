import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import csv

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    print("Please Use PyQt5!")

from libs.main import *
from libs import *
from myqlanet import *

#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class MyQLaGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.filenames = ''
        self.filename_train = ''
        self.current_idx = 0
        self.total_idx = 0
        self.total_predict_idx = 0
        self.current_predict_idx = 0
        self.valid_image_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        self.images_names = []
        self.annotation_data = []
        self.images_to_predict = None
        self.images_to_predict_names = []
        self.isPredictFolder = False
        self.setupUi(self)
        self.setFixedSize(self.size())
        pixmap = QPixmap(self.myqlanet_pixmap_path)
        self.myqlaimg_annotate.setPixmap(pixmap)
        self.myqlaimg_annotate.setScaledContents(True)
        self.myqlaimg_predict.setPixmap(pixmap)
        self.myqlaimg_predict.setScaledContents(True)
        self.myqlaimg_train.setPixmap(pixmap)
        self.myqlaimg_train.setScaledContents(True)
        self.openfile_annotate_button.clicked.connect(self.getfileAnnotate)
        self.openfile_predict_button.clicked.connect(self.getfilePredict)
        self.openfile_train_button.clicked.connect(self.getfileTrain)
        self.about_img.setPixmap(pixmap)
        self.about_img.setScaledContents(True)
        self.prev_annotate_button.clicked.connect(self.prev_pressed)
        self.next_annotate_button.clicked.connect(self.next_pressed)
        self.predict_button.clicked.connect(self.predict)
        self.next_predict_button.setVisible(self.isPredictFolder)
        self.prev_predict_button.setVisible(self.isPredictFolder)
        self.next_predict_button.clicked.connect(self.next_predict)
        self.prev_predict_button.clicked.connect(self.prev_predict)
        self.total_predict.setVisible(self.isPredictFolder)
        self.current_predict.setVisible(self.isPredictFolder)
        self.train_button.clicked.connect(self.train)
        self.set_annotate_button.clicked.connect(self.set_annotate)
        self.save_annotate_button.clicked.connect(self.save_annotate)
        self.save_predict_button.clicked.connect(self.save_predict)
        self.save_train_button.clicked.connect(self.save_train)
        self.annotate_canvas_img.setActive(False)
        self.metadata_path = ''
        self.images = []
        self.target_size = (581, 441)
        self.error_tab_idx = (0, 2)
        
        #MyQLaNet tools
        self.myqlanet = MyQLaNet()
        self.image_adjustment = ImageAdjustment()
        self.dataset_adjustment = DatasetAdjustment()

    def prev_predict(self):
        if(self.isPredictFolder):
            self.current_predict_idx -= 1
            if(self.current_predict_idx < 0):
                self.current_predict_idx = 0
            self.current_predict.setText("Current : " + str(self.current_predict_idx + 1) + '/' + str(self.total_predict_idx + 1))
            if (len(self.images_to_predict_names) > 0):
                height, width, channel = self.images_to_predict[self.current_predict_idx].shape
                bytesPerLine = 3 * width
                qImg = QImage(self.images_to_predict[self.current_predict_idx].data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                self.imshow_predict.setPixmap(pixmap)
                self.imshow_predict.setScaledContents(True)

    def next_predict(self):
        if(self.isPredictFolder):
            self.current_predict_idx += 1
            if(self.current_predict_idx > self.total_predict_idx):
                self.current_predict_idx = self.total_predict_idx
            self.current_predict.setText("Current : " + str(self.current_predict_idx + 1) + '/' + str(self.total_predict_idx + 1))
            if (len(self.images_to_predict_names) > 0):
                height, width, channel = self.images_to_predict[self.current_predict_idx].shape
                bytesPerLine = 3 * width
                qImg = QImage(self.images_to_predict[self.current_predict_idx].data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                self.imshow_predict.setPixmap(pixmap)
                self.imshow_predict.setScaledContents(True)

    def save_predict(self):
        dlg = AlertDialog(self)
        dlg.setStatus('openfile_issue')
        dlg.exec_()

    def save_train(self):
       dlg = AlertDialog(self)
       dlg.setStatus('openfile_issue')
       dlg.exec_()

    def set_annotate(self):
        try:
            img = self.images[self.current_idx]
            img_size = (img.shape[0], img.shape[1])
            scaledX = img_size[0]/self.target_size[0]
            scaledY = img_size[1]/self.target_size[1]
            annotation_box = self.annotate_canvas_img.getRect()
            point = [annotation_box[0], annotation_box[1]]
            wh = [annotation_box[2], annotation_box[3]]
            if(wh[0] < 0 and wh[1] < 0):
               point[0] += wh[0]
               point[1] += wh[1]
               wh[0] = abs(wh[0])
               wh[1] = abs(wh[1])
            elif(wh[0] < 0 and wh[1] > 0):
               point[0] += wh[0]
               wh[0] = abs(wh[0])
            elif(wh[0] > 0 and wh[1] < 0):
               point[1] += wh[1]
               wh[1] = abs(wh[1])
            point[0] *= scaledX
            point[0] = int(point[0])
            point[1] *= scaledY
            point[1] = int(point[1])
            wh[0] *= scaledX
            wh[0] = int(wh[0])
            wh[1] *= scaledY
            wh[1] = int(wh[1])
            endpoint = (point[0] + wh[0], point[1] + wh[1])
        
            self.annotation_data[self.current_idx] = str( str(self.current_idx) + ', ' + self.images_names[self.current_idx] + ', ' + str(endpoint[1]) + ', ' + str(endpoint[0]) + ', ' + str(point[1]) + ', ' + str(point[0]) + '\n')
        except:
            dlg = AlertDialog(self)
            dlg.setStatus('openfile_issue')
            dlg.exec_()

    def save_annotate(self):
        for data in self.annotation_data:
            csv_file = open(self.metadata_path, "a")
            csv_file.write(data)
            csv_file.close()
        if(len(self.annotation_data)>0):
            print("Annotation Data Saved!")
            dlg = AlertDialog(self)
            dlg.setStatus('saved')
            dlg.exec_()
        else:
            dlg = AlertDialog(self)
            dlg.setStatus('openfile_issue')
            dlg.exec_()

    def train(self):
        if(self.filename_train == ''):
            dlg = AlertDialog(self)
            dlg.setStatus('openfile_issue')
            dlg.exec_()
            return None
        dlg = AlertDialog(self)
        dlg.setStatus('train_not_found')
        missing = True
        if(self.isPredictFolder):
            name = os.path.join(self.filename_train, "annotation.csv")
            if(os.path.exists(name)):
                missing = False
        else:
            name = os.path.dirname(os.path.abspath(self.filename_train))
            name = os.path.join(name, "annotation.csv")
            if(os.path.exists(name)):
                missing = False
        if (missing):
            dlg.exec_()
            self.tabWidget.setCurrentIndex(self.error_tab_idx[0])
        else:
            print("Yeet!")

    def predict(self):
        if(self.filenames == ''):
            dlg = AlertDialog(self)
            dlg.setStatus('openfile_issue')
            dlg.exec_()
            return None
        dlg = AlertDialog(self)
        dlg.setStatus('predict_not_found')
        missing = True
        result = None
        weight_path = ''
        if(self.isPredictFolder):
            weight_path = os.path.join(self.filenames, "weight.pth")
            if(os.path.exists(weight_path)):
                missing = False
        else:
            weight_path = os.path.dirname(os.path.abspath(self.filenames))
            weight_path = os.path.join(weight_path, "weight.pth")
            if(os.path.exists(weight_path)):
                missing = False
        if (missing):
            dlg.exec_()
            self.tabWidget.setCurrentIndex(self.error_tab_idx[1])
        else:
            result_path = ''
            result_csv_path = ''
            if(self.isPredictFolder):
                #dataset_path = os.path.join(self.filenames, "annotation.csv")
                #dataset = MaculaDataset(dataset_path,self.filenames)
                #self.myqlanet.compile((dataset,dataset))
                result = self.myqlanet.predict(weight_path, self.filenames)
                result_path = os.path.join(self.filenames, "result")
                try:
                    os.mkdir(result_path)
                    print("Directory " , result_path ,  " Created ") 
                except FileExistsError:
                    print("Directory " , result_path ,  " already exists")
                result_csv_path = os.path.join(self.filenames, "result/result.csv")
            else:
                root_path = os.path.dirname(os.path.abspath(self.filenames))
                #dataset_path = os.path.join(root_path, "annotation.csv")
                #dataset = MaculaDataset(dataset_path, root_path)
                #self.myqlanet.compile((dataset))
                result = self.myqlanet.predict(weight_path, root_path)
                result_path = os.path.join(root_path, "result")
                try:
                    os.mkdir(result_path)
                    print("Directory " , result_path ,  " Created ") 
                except FileExistsError:
                    print("Directory " , result_path ,  " already exists")
                result_csv_path = os.path.join(root_path, "result/result.csv")
            if(result == None):
                dlg.exec_()
                self.tabWidget.setCurrentIndex(self.error_tab_idx[1])
                return None
            self.annotation_data = result
            csv_file = open(result_csv_path, "w")
            csv_file.write('img_name, y_lower, x_lower, y_upper, x_upper' + '\n')
            csv_file.close() 
            for idx, bbox in enumerate(self.annotation_data):
                start_point = (int(bbox[3]), int(bbox[2]))
                end_point = (int(bbox[1]), int(bbox[0]))
                color = (0, 255, 0)
                thickness = 4
                self.images_to_predict[idx] = cv2.rectangle(self.images_to_predict[idx], start_point, end_point, color, thickness)
                image_name = os.path.join(result_path, self.images_to_predict_names[idx])
                cv2.imwrite(image_name, self.images_to_predict[idx])
                data = str(str(self.images_to_predict_names[idx]) + ', ' + str(end_point[1]) + ', ' + str(end_point[0]) + ', ' + str(start_point[1]) + ', ' + str(end_point[0]))
                csv_file = open(result_csv_path, "a")
                csv_file.write(data)
                csv_file.close()
                

    def prev_pressed(self):
        self.current_idx -= 1
        if(self.current_idx < 0):
            self.current_idx = 0
        else:
            self.annotate_canvas_img.deleteRect()
        self.current_annotate.setText("Current : " + str(self.current_idx + 1) + '/' + str(self.total_idx + 1))
        if (len(self.images_names) > 0):
            height, width, channel = self.images[self.current_idx].shape
            bytesPerLine = 3 * width
            qImg = QImage(self.images[self.current_idx].data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.annotate_img.setPixmap(pixmap)
            self.annotate_img.setScaledContents(True)

    def next_pressed(self):
        self.current_idx += 1
        if(self.current_idx > self.total_idx):
            self.current_idx = self.total_idx
        else:
            self.annotate_canvas_img.deleteRect()
        self.current_annotate.setText("Current : " + str(self.current_idx + 1) + '/' + str(self.total_idx + 1))
        if (len(self.images_names) > 0):
            height, width, channel = self.images[self.current_idx].shape
            bytesPerLine = 3 * width
            qImg = QImage(self.images[self.current_idx].data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.annotate_img.setPixmap(pixmap)
            self.annotate_img.setScaledContents(True)

    def getfileAnnotate(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
	
        if dlg.exec_():
            self.annotate_canvas_img.setActive(True)
            self.images_names = []
            self.annotation_data = []
            self.total_idx = 0
            filenames = dlg.selectedFiles()
            filenames = filenames[0]
            self.metadata_path = os.path.join(filenames, "annotation.csv")
            if(os.path.exists(self.metadata_path)):
                os.remove(self.metadata_path)
            csv_file = open(self.metadata_path, "w")
            csv_file.write('img_name, y_lower, x_lower, y_upper, x_upper' + '\n')
            csv_file.close() 
            self.image_adjustment.setPath(filenames)
            self.images = self.image_adjustment.getResult()
            self.images_names = self.image_adjustment.getNames()
            self.total_idx = len(self.images_names)
            self.annotation_data = self.images_names
            self.current_idx = 0
            if(self.total_idx > 0):
                self.total_idx -= 1
            self.total_annotate.setText("Total : " + str(self.total_idx + 1))
            self.current_annotate.setText("Current : " + str(self.current_idx + 1) + '/' + str(self.total_idx + 1))
            height, width, channel = self.images[self.current_idx].shape
            bytesPerLine = 3 * width
            qImg = QImage(self.images[self.current_idx].data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.annotate_img.setPixmap(pixmap)
            self.annotate_img.setScaledContents(True)

    def getfilePredict(self):
        dlg = OpenFile(self)
        dlg.exec_()
        try:
            names = dlg.getFileName()
            if names == '':
                return None
            self.filenames = names
            if(os.path.isdir(self.filenames)):
                self.images_to_predict = []
                self.images_to_predict_names = []
                self.total_predict_idx = 0
                self.isPredictFolder = True
                self.image_adjustment.setPath(self.filenames)
                self.images_to_predict = self.image_adjustment.getResult()
                self.images_to_predict_names = self.image_adjustment.getNames()
                self.total_predict_idx = len(self.images_to_predict)
                self.current_predict_idx = 0
                if(self.total_predict_idx > 0):
                    self.total_predict_idx -= 1
                self.total_predict.setText("Total : " + str(self.total_predict_idx + 1))
                self.current_predict.setText("Current : " + str(self.current_predict_idx + 1) + '/' + str(self.total_predict_idx + 1))
                height, width, channel = self.images_to_predict[self.current_predict_idx].shape
                bytesPerLine = 3 * width
                qImg = QImage(self.images_to_predict[self.current_predict_idx].data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                self.imshow_predict.setPixmap(pixmap)
                self.imshow_predict.setScaledContents(True)
            else:
                self.isPredictFolder = False
                self.image_adjustment.setPath(self.filenames)
                self.images_to_predict = self.image_adjustment.getResult()
                self.images_to_predict_names = self.image_adjustment.getNames()
                height, width, channel = self.images_to_predict.shape
                bytesPerLine = 3 * width
                qImg = QImage(self.images_to_predict.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                self.imshow_predict.setPixmap(pixmap)
                self.imshow_predict.setScaledContents(True)
        except: 
            print("Please Select the File!")
        
        self.next_predict_button.setVisible(self.isPredictFolder)
        self.prev_predict_button.setVisible(self.isPredictFolder)
        self.total_predict.setVisible(self.isPredictFolder)
        self.current_predict.setVisible(self.isPredictFolder)

    def getfileTrain(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
		
        if dlg.exec_():
            self.filenames = dlg.selectedFiles()
            self.filenames = self.filenames[0]

def create_app(argv=[]):
    app = QApplication(argv)
    win = MyQLaGUI()                    
    win.show()
    return app, win

def main():
    app, _win = create_app(sys.argv)                      
    return app.exec_()               

if __name__ == '__main__':              
    sys.exit(main())
