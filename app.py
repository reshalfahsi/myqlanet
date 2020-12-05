from myqlanet.network.predict import predict
import sys
import os
import numpy as np
import cv2
import pandas as pd
import csv
import time

import threading

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    print("Please Use PyQt5!")

from libs.main import *
from libs import *
from myqlanet import *


class MyQLaGUI(QMainWindow, Ui_MainWindow):

    train_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.filenames = ''
        self.filenames_train = ''
        self.current_idx = 0
        self.total_idx = 0
        self.total_predict_idx = 0
        self.current_predict_idx = 0
        self.valid_image_extensions = VALID_IMAGE_FORMATS
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
        self.plt = self.lossPlot.figure.subplots()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        self.list_loss = []
        self.list_loss_copy = []
        self.isTrain = False
        self.isTrainSuccess = False
        self.epoch_now = -1
        self.last_epoch = -1
        self.iou_now = 0
        self.max_epoch = 0

        self.bbox_annotation = []
        self.annotation_df = None
        self.bbox_prediction = {}
        self.predict_df = None
        self.predict_csv_file = ''
        self.images_predicted_names = []

        self.weight_path = ''
        self.train_thread = threading.Thread(target=self.train_thread_fn)

        self.thread_called = False
        self.percent = 0
        self.isLastTrain = False

        self.train_signal.connect(self.training_process)

        # MyQLaNet tools
        self.myqlanet = MyQLaNet()
        self.image_adjustment = ImageAdjustment()
        self.dataset_adjustment = DatasetAdjustment()

    def train_thread_fn(self):
        if self.isTrain:
            self.isTrainSuccess = self.myqlanet.fit(self.weight_path)
            self.isTrain = False
            self.training_status_label.setText("Training Progress : Finished")
        self.thread_called = True

    @pyqtSlot()
    def training_process(self):
        self.isTrain = True

        self.max_epoch = self.myqlanet.max_epoch()

        if not self.train_thread.is_alive() and not self.thread_called:
            self.train_thread.start()
        else:
            self.train_thread.join()
            self.train_thread = None
            self.train_thread = threading.Thread(target=self.train_thread_fn)
            self.train_thread.start()

    def update(self):
        self.plt.clear()

        x_lim = self.plt.get_xlim()
        y_lim = self.plt.get_ylim()
        x_text = x_lim[0] + ((x_lim[1] - x_lim[0])/64)
        y_text = y_lim[1] - ((y_lim[1] - y_lim[0])/5)

        if(self.isTrain):
            if not self.isLastTrain:
                self.epoch_now = -1
                self.last_epoch = -1
                self.list_loss = []
                self.isLastTrain = True

            self.epoch_now, loss = self.myqlanet.update_loss()
            _, self.iou_now = self.myqlanet.update_iou()

            if(self.epoch_now > self.last_epoch):
                self.list_loss.append(loss)
                self.last_epoch = self.epoch_now
            elif(self.last_epoch >= 0):
                self.list_loss[self.last_epoch] = loss
            index_loss = range(self.epoch_now + 1)
            self.plt.plot(index_loss, self.list_loss, '-r', label="loss")
            self.plt.plot([], [], ' ', label=str(
                'IOU: {:.3f}'.format(round(self.iou_now, 3))))
            self.plt.legend(loc='upper left')
            self.percent = float(
                float(self.epoch_now + 1.0) / float(self.max_epoch)) * 100.0

            self.training_status_label.setText(
                "Training Progress : On Progress")
            self.list_loss_copy = self.list_loss
        else:

            if self.thread_called:

                index_loss = range(self.epoch_now + 1)
                self.plt.plot(index_loss, self.list_loss_copy,
                              '-r', label="loss")
                self.plt.plot([], [], ' ', label=str(
                    'IOU: {:.3f}'.format(round(self.iou_now, 3))))
                self.plt.legend(loc='upper left')
                if self.isLastTrain:
                    self.epoch_now, loss = self.myqlanet.update_loss()
                    self.list_loss_copy.append(loss)
                    self.percent = float(
                        float(self.epoch_now + 1.0) / float(self.max_epoch)) * 100.0

            else:
                self.plt.text(x_text, y_text, str(
                    'IOU: {:.3f}'.format(round(self.iou_now, 3))), fontsize=10)
                self.epoch_now = -1
                self.last_epoch = -1
                self.list_loss = []
            self.isLastTrain = False

        self.training_progress.setValue(self.percent)

        self.plt.figure.canvas.draw()

    def prev_predict(self):
        if(self.isPredictFolder):
            self.current_predict_idx -= 1
            if(self.current_predict_idx < 0):
                self.current_predict_idx = 0
            self.current_predict.setText(
                "Current : " + str(self.current_predict_idx + 1) + '/' + str(self.total_predict_idx + 1))
            if (len(self.images_to_predict_names) > 0):
                height, width, channel = self.images_to_predict[self.current_predict_idx].shape
                bytesPerLine = 3 * width
                qImg = QImage(self.images_to_predict[self.current_predict_idx].data,
                              width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                self.imshow_predict.setPixmap(pixmap)
                self.imshow_predict.setScaledContents(True)

    def next_predict(self):
        if(self.isPredictFolder):
            self.current_predict_idx += 1
            if(self.current_predict_idx > self.total_predict_idx):
                self.current_predict_idx = self.total_predict_idx
            self.current_predict.setText(
                "Current : " + str(self.current_predict_idx + 1) + '/' + str(self.total_predict_idx + 1))
            if (len(self.images_to_predict_names) > 0):
                height, width, channel = self.images_to_predict[self.current_predict_idx].shape
                bytesPerLine = 3 * width
                qImg = QImage(self.images_to_predict[self.current_predict_idx].data,
                              width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                self.imshow_predict.setPixmap(pixmap)
                self.imshow_predict.setScaledContents(True)

    def save_predict(self):

        self.predict_df.to_csv(self.predict_csv_file, index=False)
        print("Prediction Data Saved!")
        dlg = AlertDialog(self)
        dlg.setStatus('saved_prediction')
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
            bbox = (point[1] + wh[1], point[0] + wh[0],
                    point[1], point[0])  # y+h,x+w,y,x
            point[0] *= scaledX
            point[0] = int(point[0])
            point[1] *= scaledY
            point[1] = int(point[1])
            wh[0] *= scaledX
            wh[0] = int(wh[0])
            wh[1] *= scaledY
            wh[1] = int(wh[1])
            endpoint = (point[0] + wh[0], point[1] + wh[1])

            self.bbox_annotation[self.current_idx] = bbox
            self.annotation_df.iloc[self.current_idx, 1:] = np.array(
                [endpoint[1], endpoint[0], point[1], point[0]])
            # else:
            #     self.annotation_data[self.current_idx] = str(str(self.current_idx) + ', ' + self.images_names[self.current_idx] + ', ' + str(
            #         endpoint[1]) + ', ' + str(endpoint[0]) + ', ' + str(point[1]) + ', ' + str(point[0]) + '\n')
        except:
            dlg = AlertDialog(self)
            dlg.setStatus('openfile_issue')
            dlg.exec_()

    def save_annotate(self):

        self.annotation_df.to_csv(self.metadata_path, index=False)
        print("Annotation Data Saved!")
        dlg = AlertDialog(self)
        dlg.setStatus('saved')
        dlg.exec_()

    def train(self):
        if(self.filenames_train == ''):
            dlg = AlertDialog(self)
            dlg.setStatus('openfile_issue')
            dlg.exec_()
            return None
        dlg = AlertDialog(self)
        dlg.setStatus('train_not_found')
        missing = True
        dataset_path = os.path.join(self.filenames_train, "annotation.csv")
        self.weight_path = os.path.join(self.filenames_train, "weight.pth")
        if(os.path.exists(self.filenames_train)):
            missing = False
        if (missing):
            dlg.exec_()
            self.tabWidget.setCurrentIndex(self.error_tab_idx[0])
        else:
            dataset = MaculaDataset(dataset_path, self.filenames_train)
            self.myqlanet.compile(dataset)
            self.train_signal.emit()

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
        # if(self.isPredictFolder):
        weight_path = os.path.join(self.filenames, "weight.pth")
        if(os.path.exists(weight_path)):
            missing = False
        # else:
        #    weight_path = os.path.dirname(os.path.abspath(self.filenames))
        #    weight_path = os.path.join(weight_path, "weight.pth")
        #    if(os.path.exists(weight_path)):
        #        missing = False
        if (missing):
            dlg.exec_()
            self.tabWidget.setCurrentIndex(self.error_tab_idx[1])
        else:
            path = os.path.join(
                self.filenames, self.images_to_predict_names[self.current_predict_idx])
            print(path)
            result = self.myqlanet.predict(weight_path, path)
            # print(result)

            dummy_df = self.predict_df.set_index('img_name')

            if (self.images_to_predict_names[self.current_predict_idx] in dummy_df.index):
                idx = self.predict_df.index[self.predict_df['img_name'] ==
                                            self.images_to_predict_names[self.current_predict_idx]].tolist()
                idx = idx[0]
                self.predict_df.iloc[idx, 1:] = np.array(
                    [int(result[0]), int(result[1]), int(result[2]), int(result[3])])
            else:
                new_predict = {'img_name': self.images_to_predict_names[self.current_predict_idx],
                               'y_lower': int(result[0]), 'x_lower': int(result[1]), 'y_upper': int(result[2]), 'x_upper': int(result[3])}
                self.predict_df = self.predict_df.append(
                    new_predict, ignore_index=True)

            if(self.images_to_predict_names[self.current_predict_idx] not in self.images_predicted_names):
                self.images_predicted_names.append(
                    self.images_to_predict_names[self.current_predict_idx])

            crop = CropImage()
            resize = ResizeImage()
            image_name = os.path.join(
                self.filenames, self.images_to_predict_names[self.current_predict_idx])
            image = cv2.imread(image_name)
            orig_size = (image.shape[0], image.shape[1])
            image = crop.run(image)
            resize.process(image, VALID_IMAGE_SIZE)
            image = resize.getResult()
            image_name = os.path.join(self.filenames, str(
                'result/' + self.images_to_predict_names[self.current_predict_idx]))
            start_point = (int(result[3]), int(result[2]))
            end_point = (int(result[1]), int(result[0]))
            # print(start_point, end_point)
            color = (0, 255, 0)
            thickness = 5
            image = cv2.rectangle(image, start_point,
                                  end_point, color, thickness)
            # resize.process(image, orig_size)
            # image = resize.getResult()
            cv2.imwrite(image_name, image)
            # print(image_name)
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine,
                          QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.imshow_predict.setPixmap(pixmap)
            self.imshow_predict.setScaledContents(True)

    def prev_pressed(self):
        self.current_idx -= 1
        if(self.current_idx < 0):
            self.current_idx = 0
        else:
            self.annotate_canvas_img.deleteRect()
        self.current_annotate.setText(
            "Current : " + str(self.current_idx + 1) + '/' + str(self.total_idx + 1))
        if self.bbox_annotation[self.current_idx] is not None:
            rect = (self.bbox_annotation[self.current_idx][3], self.bbox_annotation[self.current_idx]
                    [2], self.bbox_annotation[self.current_idx][1], self.bbox_annotation[self.current_idx][0])
            self.annotate_canvas_img.setRect(rect)
        if (len(self.images_names) > 0):
            height, width, channel = self.images[self.current_idx].shape
            bytesPerLine = 3 * width
            qImg = QImage(self.images[self.current_idx].data, width,
                          height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.annotate_img.setPixmap(pixmap)
            self.annotate_img.setScaledContents(True)

    def next_pressed(self):
        self.current_idx += 1
        if(self.current_idx > self.total_idx):
            self.current_idx = self.total_idx
        else:
            self.annotate_canvas_img.deleteRect()
        self.current_annotate.setText(
            "Current : " + str(self.current_idx + 1) + '/' + str(self.total_idx + 1))
        if self.bbox_annotation[self.current_idx] is not None:
            rect = (self.bbox_annotation[self.current_idx][3], self.bbox_annotation[self.current_idx]
                    [2], self.bbox_annotation[self.current_idx][1], self.bbox_annotation[self.current_idx][0])
            self.annotate_canvas_img.setRect(rect)
        if (len(self.images_names) > 0):
            height, width, channel = self.images[self.current_idx].shape
            bytesPerLine = 3 * width
            qImg = QImage(self.images[self.current_idx].data, width,
                          height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.annotate_img.setPixmap(pixmap)
            self.annotate_img.setScaledContents(True)

    def getfileAnnotate(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)

        bbox_start = None

        if dlg.exec_():
            self.annotate_canvas_img.setActive(True)
            self.images_names = []
            self.annotation_data = []
            self.total_idx = 0
            filenames = dlg.selectedFiles()
            filenames = filenames[0]

            self.image_adjustment.setPath(filenames)
            self.images = self.image_adjustment.getResult()
            self.images_names = self.image_adjustment.getNames()
            self.total_idx = len(self.images_names)
            self.annotation_data = self.images_names
            self.bbox_annotation = len(self.images_names) * [None]

            self.metadata_path = os.path.join(filenames, "annotation.csv")
            if(os.path.exists(self.metadata_path)):
                self.annotation_df = pd.read_csv(
                    self.metadata_path, skipinitialspace=True)
                bboxes = self.annotation_df.iloc[:, 1:]
                bboxes = bboxes.to_numpy().astype(np.int32)
                for idx, bbox in enumerate(bboxes):
                    img_size = self.images[idx].shape
                    scaledX = self.target_size[0]/img_size[0]
                    scaledY = self.target_size[1]/img_size[1]
                    self.bbox_annotation[idx] = (
                        bbox[0] * scaledY, bbox[1] * scaledX, bbox[2] * scaledY, bbox[3] * scaledX)
                    if idx == 0:
                        bbox_start = self.bbox_annotation[idx]
            else:
                data = {'img_name': [img for img in self.images_names], 'y_lower': [idx for idx, _ in enumerate(self.images_names)], 'x_lower': [idx for idx, _ in enumerate(
                    self.images_names)], 'y_upper': [idx for idx, _ in enumerate(self.images_names)], 'x_upper': [idx for idx, _ in enumerate(self.images_names)]}
                self.annotation_df = pd.DataFrame(data)

            self.current_idx = 0
            if(self.total_idx > 0):
                self.total_idx -= 1
            self.total_annotate.setText("Total : " + str(self.total_idx + 1))
            self.current_annotate.setText(
                "Current : " + str(self.current_idx + 1) + '/' + str(self.total_idx + 1))
            height, width, channel = self.images[self.current_idx].shape
            bytesPerLine = 3 * width
            qImg = QImage(self.images[self.current_idx].data, width,
                          height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.annotate_img.setPixmap(pixmap)
            self.annotate_img.setScaledContents(True)
            if bbox_start is not None:
                rect = (self.bbox_annotation[0][3], self.bbox_annotation[0]
                        [2], self.bbox_annotation[0][1], self.bbox_annotation[0][0])
                self.annotate_canvas_img.setRect(rect)

    def getfilePredict(self):

        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)

        if dlg.exec_():
            self.filenames = dlg.selectedFiles()
            self.filenames = self.filenames[0]
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
                self.total_predict.setText(
                    "Total : " + str(self.total_predict_idx + 1))
                self.current_predict.setText(
                    "Current : " + str(self.current_predict_idx + 1) + '/' + str(self.total_predict_idx + 1))
                height, width, channel = self.images_to_predict[self.current_predict_idx].shape
                bytesPerLine = 3 * width
                qImg = QImage(self.images_to_predict[self.current_predict_idx].data,
                              width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                self.imshow_predict.setPixmap(pixmap)
                self.imshow_predict.setScaledContents(True)

                result_path = os.path.join(self.filenames, "result")
                self.predict_csv_file = os.path.join(
                    result_path, "prediction.csv")
                # bbox_start = None

                try:
                    os.mkdir(result_path)
                    print("Directory ", result_path,  " Created ")

                    data = {'img_name': [], 'y_lower': [],
                            'x_lower': [], 'y_upper': [], 'x_upper': []}
                    self.predict_df = pd.DataFrame(data)

                except FileExistsError:
                    print("Directory ", result_path,  " already exists")
                    if(os.path.exists(self.predict_csv_file)):
                        self.predict_df = pd.read_csv(
                            self.predict_csv_file, skipinitialspace=True)
                        bboxes = self.predict_df.iloc[:, 1:]
                        predict_data = self.predict_df.iloc[:, 0].to_numpy()
                        bboxes = bboxes.to_numpy().astype(np.int32)
                        for idx, bbox in enumerate(bboxes):
                            img_size = self.images_to_predict[idx].shape
                            scaledX = self.target_size[0]/img_size[0]
                            scaledY = self.target_size[1]/img_size[1]
                            self.images_predicted_names.append(
                                predict_data[idx])
                            self.bbox_prediction[predict_data[idx]] = (
                                bbox[0] * scaledY, bbox[1] * scaledX, bbox[2] * scaledY, bbox[3] * scaledX)
                    else:
                        data = {'img_name': [], 'y_lower': [],
                                'x_lower': [], 'y_upper': [], 'x_upper': []}
                        self.predict_df = pd.DataFrame(data)

        self.next_predict_button.setVisible(self.isPredictFolder)
        self.prev_predict_button.setVisible(self.isPredictFolder)
        self.total_predict.setVisible(self.isPredictFolder)
        self.current_predict.setVisible(self.isPredictFolder)

    def getfileTrain(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)

        if dlg.exec_():
            self.filenames_train = dlg.selectedFiles()
            self.filenames_train = self.filenames_train[0]


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
