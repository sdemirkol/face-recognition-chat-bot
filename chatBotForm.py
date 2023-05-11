from PyQt5.QtWidgets import *
from chatBot import Ui_chatBotWindow

class chatBotForm(QMainWindow):
    def __init__(self) -> None:
        super().__init__()        
        self.chatBotFormPage = Ui_chatBotWindow()
        self.chatBotFormPage.setupUi(self)
        self.chatBotFormPage.ageValueLabel.setText("Age Label")