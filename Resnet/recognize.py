#coding:utf-8
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui
import classify
class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setObjectName("widget")
        self.resize(490, 506)
        self.setMinimumSize(QtCore.QSize(100, 100))
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.gridLayoutWidget = QtWidgets.QWidget(self)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(60, 120, 381, 301))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(70, 50, 54, 20))
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(self)
        self.textEdit.setGeometry(QtCore.QRect(120, 45, 261, 25))
        self.textEdit.setObjectName("textEdit")
        self.toolButton = QtWidgets.QToolButton(self)
        self.toolButton.setGeometry(QtCore.QRect(379, 43, 50, 28))
        self.toolButton.setObjectName("toolButton")
        self.toolButton.clicked.connect(self.msg)
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(200, 80, 81, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.sbing)
        #  放图片的label
        self.label2 = QtWidgets.QLabel(self)
        self.label2.setGeometry(QtCore.QRect(72, 150, 360, 300))
        #  参数分别是左上点距左边框宽度，距顶高度，长度，高度
        self.label2.setObjectName("label2")

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)
    def retranslateUi(self, widget):
        _translate = QtCore.QCoreApplication.translate
        widget.setWindowTitle(_translate("widget", "CNN图片识别 by White-Lotus"))
        self.label.setText(_translate("widget", "源图片:"))
        self.toolButton.setText(_translate("widget", "浏览"))
        self.pushButton.setText(_translate("widget", "开始识别"))
    def msg(self): 

        fileName1, filetype = QFileDialog.getOpenFileName(self,
                                                          "选取文件",
                                                          "/",
                                                          "All Files (*);;image Files (*.jpg)")  # 设置文件扩展名过滤,注意用双分号间隔
        #  print(fileName1, filetype)
        #  print(fileName1)
 
        png = QtGui.QPixmap(fileName1).scaled(self.label2.width(), self.label2.height())
        self.label2.setPixmap(png)
        self.textEdit.setText(fileName1)
        classify.imgf=fileName1
    def sbing(self):
        self.pushButton.setText("识别中")
        classify.sjsy()
        self.pushButton.setText("开始识别")
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
    exit()
