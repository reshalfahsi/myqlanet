# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from . import canvas

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.myqlanet_pixmap_path = "resources/icons/MyQLaNet.png"
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(817, 599)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(self.myqlanet_pixmap_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 811, 551))
        self.tabWidget.setObjectName("tabWidget")
        self.annotate_tab = QtWidgets.QWidget()
        self.annotate_tab.setObjectName("annotate_tab")
        self.set_annotate_button = QtWidgets.QPushButton(self.annotate_tab)
        self.set_annotate_button.setGeometry(QtCore.QRect(660, 90, 89, 25))
        self.set_annotate_button.setObjectName("set_annotate_button")
        self.prev_annotate_button = QtWidgets.QPushButton(self.annotate_tab)
        self.prev_annotate_button.setGeometry(QtCore.QRect(610, 140, 89, 25))
        self.prev_annotate_button.setObjectName("prev_annotate_button")
        self.next_annotate_button = QtWidgets.QPushButton(self.annotate_tab)
        self.next_annotate_button.setGeometry(QtCore.QRect(710, 140, 89, 25))
        self.next_annotate_button.setObjectName("next_annotate_button")
        self.openfile_annotate_button = QtWidgets.QPushButton(self.annotate_tab)
        self.openfile_annotate_button.setGeometry(QtCore.QRect(660, 220, 89, 25))
        self.openfile_annotate_button.setObjectName("openfile_annotate_button")
        self.total_annotate = QtWidgets.QLabel(self.annotate_tab)
        self.total_annotate.setGeometry(QtCore.QRect(30, 490, 101, 17))
        self.total_annotate.setObjectName("total_annotate")
        self.current_annotate = QtWidgets.QLabel(self.annotate_tab)
        self.current_annotate.setGeometry(QtCore.QRect(140, 490, 161, 17))
        self.current_annotate.setObjectName("current_annotate")
        self.save_annotate_button = QtWidgets.QPushButton(self.annotate_tab)
        self.save_annotate_button.setGeometry(QtCore.QRect(660, 250, 89, 25))
        self.save_annotate_button.setObjectName("save_annotate_button")
        self.myqlaimg_annotate = QtWidgets.QLabel(self.annotate_tab)
        self.myqlaimg_annotate.setGeometry(QtCore.QRect(650, 350, 141, 131))
        self.myqlaimg_annotate.setObjectName("myqlaimg_annotate")
        self.annotate_img = QtWidgets.QLabel(self.annotate_tab)
        self.annotate_img.setGeometry(QtCore.QRect(20, 30, 581, 441))
        self.annotate_img.setText("")
        self.annotate_img.setObjectName("annotate_img")
        self.annotate_canvas_img = canvas.Canvas(self.annotate_tab)
        self.annotate_canvas_img.setGeometry(QtCore.QRect(20, 30, 581, 441))
        self.annotate_canvas_img.setText("")
        self.annotate_canvas_img.setObjectName("annotate_canvas_img")
        self.tabWidget.addTab(self.annotate_tab, "")
        self.predict_tab = QtWidgets.QWidget()
        self.predict_tab.setObjectName("predict_tab")
        self.predict_button = QtWidgets.QPushButton(self.predict_tab)
        self.predict_button.setGeometry(QtCore.QRect(670, 70, 89, 25))
        self.predict_button.setObjectName("predict_button")
        self.openfile_predict_button = QtWidgets.QPushButton(self.predict_tab)
        self.openfile_predict_button.setGeometry(QtCore.QRect(670, 100, 89, 25))
        self.openfile_predict_button.setObjectName("openfile_predict_button")
        self.save_predict_button = QtWidgets.QPushButton(self.predict_tab)
        self.save_predict_button.setGeometry(QtCore.QRect(670, 130, 89, 25))
        self.save_predict_button.setObjectName("save_predict_button")
        self.myqlaimg_predict = QtWidgets.QLabel(self.predict_tab)
        self.myqlaimg_predict.setGeometry(QtCore.QRect(650, 350, 141, 131))
        self.myqlaimg_predict.setObjectName("myqlaimg_predict")
        self.imshow_predict = QtWidgets.QLabel(self.predict_tab)
        self.imshow_predict.setGeometry(QtCore.QRect(30, 30, 581, 441))
        self.imshow_predict.setText("")
        self.imshow_predict.setObjectName("imshow_predict")
        self.current_predict = QtWidgets.QLabel(self.predict_tab)
        self.current_predict.setGeometry(QtCore.QRect(140, 490, 161, 17))
        self.current_predict.setObjectName("current_predict")
        self.total_predict = QtWidgets.QLabel(self.predict_tab)
        self.total_predict.setGeometry(QtCore.QRect(30, 490, 101, 17))
        self.total_predict.setObjectName("total_predict")
        self.next_predict_button = QtWidgets.QPushButton(self.predict_tab)
        self.next_predict_button.setGeometry(QtCore.QRect(460, 480, 89, 25))
        self.next_predict_button.setObjectName("next_predict_button")
        self.prev_predict_button = QtWidgets.QPushButton(self.predict_tab)
        self.prev_predict_button.setGeometry(QtCore.QRect(360, 480, 89, 25))
        self.prev_predict_button.setObjectName("prev_predict_button")
        self.tabWidget.addTab(self.predict_tab, "")
        self.train_tab = QtWidgets.QWidget()
        self.train_tab.setObjectName("train_tab")
        self.myqlaimg_train = QtWidgets.QLabel(self.train_tab)
        self.myqlaimg_train.setGeometry(QtCore.QRect(650, 350, 141, 131))
        self.myqlaimg_train.setObjectName("myqlaimg_train")
        self.training_progress = QtWidgets.QProgressBar(self.train_tab)
        self.training_progress.setGeometry(QtCore.QRect(20, 440, 331, 31))
        self.training_progress.setProperty("value", 0)
        self.training_progress.setObjectName("training_progress")
        self.training_status_label = QtWidgets.QLabel(self.train_tab)
        self.training_status_label.setGeometry(QtCore.QRect(20, 480, 251, 20))
        self.training_status_label.setObjectName("training_status_label")
        self.save_train_button = QtWidgets.QPushButton(self.train_tab)
        self.save_train_button.setGeometry(QtCore.QRect(670, 130, 89, 25))
        self.save_train_button.setObjectName("save_train_button")
        self.openfile_train_button = QtWidgets.QPushButton(self.train_tab)
        self.openfile_train_button.setGeometry(QtCore.QRect(670, 100, 89, 25))
        self.openfile_train_button.setObjectName("openfile_train_button")
        self.train_button = QtWidgets.QPushButton(self.train_tab)
        self.train_button.setGeometry(QtCore.QRect(670, 70, 89, 25))
        self.train_button.setObjectName("train_button")
        self.lossPlot = QtWidgets.QLabel(self.train_tab)
        self.lossPlot.setGeometry(QtCore.QRect(30, 40, 561, 361))
        self.lossPlot.setObjectName("lossPlot")
        self.loss_label = QtWidgets.QLabel(self.train_tab)
        self.loss_label.setGeometry(QtCore.QRect(30, 10, 67, 17))
        self.loss_label.setObjectName("loss_label")
        self.tabWidget.addTab(self.train_tab, "")
        self.about = QtWidgets.QWidget()
        self.about.setObjectName("about")
        self.about_img = QtWidgets.QLabel(self.about)
        self.about_img.setGeometry(QtCore.QRect(290, 50, 191, 181))
        self.about_img.setObjectName("about_img")
        self.label_GitHub = QtWidgets.QLabel(self.about)
        self.label_GitHub.setGeometry(QtCore.QRect(160, 287, 460, 17))
        self.label_GitHub.setAlignment(QtCore.Qt.AlignCenter)
        self.label_GitHub.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.label_GitHub.setObjectName("label_GitHub")
        self.label_AppVersion = QtWidgets.QLabel(self.about)
        self.label_AppVersion.setGeometry(QtCore.QRect(160, 241, 460, 17))
        self.label_AppVersion.setAlignment(QtCore.Qt.AlignCenter)
        self.label_AppVersion.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.label_AppVersion.setObjectName("label_AppVersion")
        self.label_7 = QtWidgets.QLabel(self.about)
        self.label_7.setGeometry(QtCore.QRect(160, 264, 460, 17))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_HomePage = QtWidgets.QLabel(self.about)
        self.label_HomePage.setGeometry(QtCore.QRect(160, 310, 460, 17))
        self.label_HomePage.setAlignment(QtCore.Qt.AlignCenter)
        self.label_HomePage.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.label_HomePage.setObjectName("label_HomePage")
        self.tabWidget.addTab(self.about, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 817, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MyQLaNet"))
        self.set_annotate_button.setText(_translate("MainWindow", "Set"))
        self.prev_annotate_button.setText(_translate("MainWindow", "Previous"))
        self.next_annotate_button.setText(_translate("MainWindow", "Next"))
        self.openfile_annotate_button.setText(_translate("MainWindow", "Open FIle"))
        self.total_annotate.setText(_translate("MainWindow", "Total: 0"))
        self.current_annotate.setText(_translate("MainWindow", "Current : 0/0"))
        self.save_annotate_button.setText(_translate("MainWindow", "Save"))
        self.myqlaimg_annotate.setText(_translate("MainWindow", "Insert Image Here"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.annotate_tab), _translate("MainWindow", "Annotate"))
        self.predict_button.setText(_translate("MainWindow", "Predict"))
        self.openfile_predict_button.setText(_translate("MainWindow", "Open File"))
        self.save_predict_button.setText(_translate("MainWindow", "Save"))
        self.myqlaimg_predict.setText(_translate("MainWindow", "Insert Image Here"))
        self.current_predict.setText(_translate("MainWindow", "Current : 0/0"))
        self.total_predict.setText(_translate("MainWindow", "Total: 0"))
        self.next_predict_button.setText(_translate("MainWindow", "Next"))
        self.prev_predict_button.setText(_translate("MainWindow", "Previous"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.predict_tab), _translate("MainWindow", "Predict"))
        self.myqlaimg_train.setText(_translate("MainWindow", "Insert Image Here"))
        self.training_status_label.setText(_translate("MainWindow", "Training Progress : Stopped"))
        self.save_train_button.setText(_translate("MainWindow", "Save"))
        self.openfile_train_button.setText(_translate("MainWindow", "Open File"))
        self.train_button.setText(_translate("MainWindow", "Train"))
        self.loss_label.setText(_translate("MainWindow", "Loss"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.train_tab), _translate("MainWindow", "Train"))
        self.about_img.setText(_translate("MainWindow", "Insert Image Here"))
        self.label_GitHub.setText(_translate("MainWindow", "<html><head/><body><p><a href=\"https://github.com/reshalfahsi/myqlanet\" target=\"_blank\"><span style=\" text-decoration: underline; color:#0000ff;\">Source codes on Github</span></a></p></body></html>"))
        self.label_AppVersion.setText(_translate("MainWindow", "V 1.0.1"))
        self.label_7.setText(_translate("MainWindow", "By : Resha Dwika Hefni Al-Fahsi, et al."))
        self.label_HomePage.setText(_translate("MainWindow", "<html><head/><body><p><a href=\"https://reshalfahsi.github.io/\" target=\"_blank\"><span style=\" text-decoration: underline; color:#0000ff;\">Home page</span></a></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.about), _translate("MainWindow", "About"))
