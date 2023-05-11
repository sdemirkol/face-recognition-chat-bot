from ast import Index
import enum
import numpy as np
import sys
import openai
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, QPoint
from panel import *
from chatBot import *
from chatBotForm import chatBotForm
from recognition import FaceRecognition

age = '0'
emotion = 'sad'
image = np.ndarray
userName=''

def faceRec():     
    global age 
    global emotion
    global image
    global userName
    # age = '25'
    # emotion = 'happy'
    # print('faceRec ' + age + ' ' + emotion)
    # show_msg_success()

    fr = FaceRecognition()  
    result = fr.run_recognition() 
    print('-----------------')  
    print(result)
    print(result['isMatching'])     
    print(result['age']) 
    print(result['emotion'])
    print(result['userName'])
    age=result['age']
    emotion=result['emotion']
    image = result['image']
    userName= result['userName']

    if result['isMatching'] == True:         
       show_msg_success()
    else: 
        show_msg_fail()

#print('dış ' + age + ' ' + emotion) 

def show_msg_success():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
  
    msg.setText("Yüz Tanıma Başarılı")
      
    msg.setWindowTitle("Tarama Sonucu")
      
    msg.setStandardButtons(QMessageBox.Ok)

    msg.buttonClicked.connect(msg_clicked_success)
    msg.exec_()

conversation = []
def chat_bot_setup(userName, age, emotion):
    if (age and emotion):
        conversation.clear()
        conversation.append({"role": "system", "content": "İnsanları çok iyi anlayabilen profesyonel, kibar ve saygılı bir yaşam koçu gibi davran. Cevapların mümkün olduğunca kısa olsun. Kesinlikle örnekler verme, ek açıklamalar yapma ve hitaplarında saygı kelimeleri kullanmayı ihmal etme. Bir kişinin adını, yaşını ve duygu durumunu söylediğimde bu kişiyi görüyormuş gibi günlük bir tavsiyede bulun."})
        conversation.append({"role": "user", "content": "{}, {}, {}".format(userName, age, emotion)})
    else:
        conversation.append({ "role": "system", "content": "Soru sormadan ve selamlama cümlesi kullanmadan daha özel bir tavsiyede bulun." })
def chat_bot_api():
    openai.api_key = "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"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content

    conversation.append({ "role": "assistant", "content": result })
    return result

def like_button_clicked():
    global chatWindow
    chat_bot_setup(False, False, False)
    advice = chat_bot_api()
    chatWindow.chatBotFormPage.listWidget.addItem(QListWidgetItem(advice))

def dislike_button_clicked():
    global chatWindow
    global age 
    global emotion
    global userName

    chatWindow.chatBotFormPage.listWidget.clear()
    chat_bot_setup(userName, age, emotion)
    advice = chat_bot_api()
    chatWindow.chatBotFormPage.listWidget.addItem(QListWidgetItem(advice))

def msg_clicked_success(): 
    global chatWindow
    global age 
    global emotion
    global image
    global userName

    print('clicked ' + age + ' ' + emotion)
    chatWindow = chatBotForm()
    # chatWindow.chatBotFormPage.listView.model()
    chatWindow.chatBotFormPage.emotionValueLabel.setText(emotion)
    chatWindow.chatBotFormPage.ageValueLabel.setText(age)
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    chatWindow.chatBotFormPage.photoLabel.setPixmap(QtGui.QPixmap(qImg))
    # chatWindow.chatBotFormPage.photoLabel.setPixmap(QtGui.QPixmap("faces/sedat.jpg"))
    # chatWindow.ageValueLabel.setText('age')
    # chatWindow.emotionLabel.setText('emotion')
    chatWindow.show()

    chat_bot_setup(userName, age, emotion)
    advice = chat_bot_api()
    chatWindow.chatBotFormPage.listWidget.addItem(QListWidgetItem(advice))
    # self.ageValueLabel.setText(_translate("chatBotWindow", "TextLabel"))
    # self.emotionValueLabel.setText(_translate("chatBotWindow", "TextLabel"))
    chatWindow.chatBotFormPage.likeButton.clicked.connect(like_button_clicked)
    chatWindow.chatBotFormPage.dislikeButton.clicked.connect(dislike_button_clicked)

def show_msg_fail():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
  
    msg.setText("Yüz Tanıma Başarısız")
      
    msg.setWindowTitle("Tarama Sonucu")
      
    msg.setStandardButtons(QMessageBox.Ok)

    msg.buttonClicked.connect(msg_clicked_fail)
    msg.exec_()

def msg_clicked_fail(msg_button):
    print(msg_button.text())


app = QApplication(sys.argv)
window = QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(window) 
window.show()

ui.startButton.clicked.connect(faceRec)
ui.openButton.clicked.connect(msg_clicked_success)

sys.exit(app.exec_())