from PyQt5 import uic

with open("panel.py","w",encoding="utf-8") as fout:
    uic.compileUi("panel.ui",fout)

with open("chatBot.py","w",encoding="utf-8") as fout:
    uic.compileUi("chatBot.ui",fout)

