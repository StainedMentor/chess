from copy import deepcopy

import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGraphicsView, \
    QGraphicsScene, QGridLayout, QTextEdit, QScrollArea, QLabel, QLineEdit, QRadioButton, QButtonGroup
from PyQt5.QtGui import QBrush, QPen, QColor
from PyQt5.QtCore import Qt,QTimer

import sys
import queue
import re

from ai import convNetwork
from constants import *
from board import Board
from history import History
from client import Client

# from network import ChessNet
# net = ChessNet()
#
# net.load_state_dict(torch.load('dict.pth'))
# net.eval()

from ai.convNetwork import ChessNet
net = ChessNet()
net.load_state_dict(torch.load('ai/dict_conv.pth'))
net.eval()





class ChessWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        self.setWindowTitle('PyQt Chess')
        self.setGeometry(100, 100, 1000, 800)
        print(vars(config).items())
        self.config = config

        self.selectedPiece = None
        self.availableObjects = []
        self.availableMoves = []
        self.isWhiteTurn = True
        self.robots = False
        self.log = ""
        self.logQueue = queue.Queue()
        self.log_field = QTextEdit()

        self.timeW = 0
        self.timeB = 0
        self.timeType = 0
        self.shouldTime = False
        self.gameFinished = False
        self.boards = []
        self.boardIndex = 0


        self.addUI()
        self.createChessboard()



        self.board = Board(self.scene)
        self.board.initPieces()
        self.boards.append(deepcopy(self.board.map))


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.handleLog)
        self.timer.start(1000)

        self.force = QTimer(self)
        self.force.timeout.connect(self.forceRefresh)
        self.force.start(100)

        self.history = History()

        self.moveQueue = queue.Queue()
        self.isServer = True
        self.initServerClient()




        self.show()

    def forceRefresh(self):
        rect = self.scene.sceneRect()
        self.scene.update(rect)
        self.scene.setSceneRect(rect)

    def initServerClient(self):
        ip = self.config.IP
        port = self.config.port
        self.client = Client(ip,port, self.moveQueue, self.logQueue, self.handleReceivedMove)
        if self.client.isClient:
            self.isServer = False

    def handleLog(self):
        while not self.logQueue.empty():
            self.log += self.logQueue.get()
            self.log_field.setPlainText(self.log)

        self.updateTime()


    def updateTime(self):
        if self.timeType == 0:
            return
        if self.isWhiteTurn:
            self.timeW -= 1
            minutes = int(self.timeW / 60)
            seconds = self.timeW % 60
            self.timeWLabel.setText("white time: " + str(minutes) + ":" + str(seconds))
            if self.timeW < 0:
                self.logQueue.put("white ran out of time\n")
        else:
            self.timeB -= 1
            minutes = int(self.timeB / 60)
            seconds = self.timeB % 60
            self.timeBLabel.setText("black time: " + str(minutes) + ":" + str(seconds))
            if self.timeB < 0:
                self.logQueue.put("black ran out of time\n")

    def addUI(self):
        w = QWidget()
        self.setCentralWidget(w)
        self.grid = QGridLayout(w)
        # robot button
        self.robotButton = QPushButton("history back", w)
        self.robotButton.clicked.connect(self.robotsStart)

        self.setCentralWidget(w)
        self.grid.addWidget(self.robotButton, 9, 1)

        # log
        self.log_field.setPlainText(self.log)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.log_field)
        self.grid.addWidget(scroll_area, 0, 1)

        # timers
        self.timeWLabel = QLabel("white time: 00:00")
        self.grid.addWidget(self.timeWLabel, 2, 1)
        self.timeBLabel = QLabel("black time: 00:00")
        self.grid.addWidget(self.timeBLabel, 3, 1)

        self.timerToggle = QPushButton("history forward", w)
        self.timerToggle.clicked.connect(self.historyForward)
        self.grid.addWidget(self.timerToggle, 8, 1)

        # input
        self.command = QLineEdit(self)
        self.command.setPlaceholderText('A0,A1')
        self.submit_command = QPushButton('Submit', self)
        self.submit_command.clicked.connect(self.readText)
        self.grid.addWidget(self.command, 4, 1)
        self.grid.addWidget(self.submit_command, 5, 1)

        # input
        self.msgBox = QLineEdit(self)
        self.msgBox.setPlaceholderText('Chat')
        self.submitMsg = QPushButton('Send', self)
        self.submitMsg.clicked.connect(self.sendChat)
        self.grid.addWidget(self.msgBox, 6, 1)
        self.grid.addWidget(self.submitMsg, 7, 1)


    def sendChat(self):
        text = self.msgBox.text()
        msg = "MSG|chat: " + text + "\n"
        self.client.send_text_message(msg)

    def handleReceivedMove(self, move):
        origin, new = self.history.universalToPos(move)

        if self.isServer:
            if self.execute(origin,new):
                print("moved")
                pass
            else:
                msg = "MSG|chat: " + "invalid" + "\n"
                self.client.send_text_message(msg)
        else:
            self.execute(origin,new)




    def timeRulesTurnEnd(self):
        if self.timeType == 1:
            self.timeW = 10
            self.timeB = 10
        elif self.timeType == 3:
            if not self.isWhiteTurn:
                self.timeW += 30
            else:
                self.timeB += 30
    def toggleTime(self):
        self.timeType += 1
        self.timeType %= 4
        self.timerToggle.setText("Toggle timer: " + gameTimes[self.timeType])

        if self.timeType == 1:
            self.timeW = 10
            self.timeB = 10
        elif self.timeType == 2:
            self.timeW = 300
            self.timeB = 300
        elif self.timeType == 3:
            self.timeW = 300
            self.timeB = 300



    def robotsStart(self):
        self.boardIndex = self.boardIndex - 1
        self.board.deletePieces()
        self.board.map = self.boards[self.boardIndex]
        self.board.update()
        self.isWhiteTurn = not self.isWhiteTurn


    def historyForward(self):
        self.boardIndex = self.boardIndex + 1
        self.board.deletePieces()

        self.board.map = self.boards[self.boardIndex]
        self.board.update()
        self.isWhiteTurn = not self.isWhiteTurn



    def readText(self):
        text = self.command.text()
        origin, new = self.parse(text)
        self.execute(origin,new)
    def execute(self, origin, new):

        if origin is None:
            self.logQueue.put("invalid format\n")
            return
        piece = self.board.map[origin[1]][origin[0]]
        if piece is None:
            self.logQueue.put("invalid move (no piece)\n")
            return

        self.availableMoves = self.board.selectPiece(origin[0],origin[1])


        if self.board.map[origin[1]][origin[0]].isWhite != self.isWhiteTurn:
            self.logQueue.put("wrong turn\n")

            return

        if origin == new or new not in self.availableMoves:
            self.logQueue.put("invalid move\n")
            return

        x1,y1,newx,newy = self.board.move(origin, new)

        self.logQueue.put("Moved from {x1},{y1} to {x2},{y2}.\n".format(x1=abc[x1],y1=str(y1),x2=abc[newx],y2=str(newy)))

        if self.board.checkCheck(self.board.map, not self.isWhiteTurn):
            if self.board.checkMate(not self.isWhiteTurn):
                self.logQueue.put("checkmate")
                self.gameFinished = True
            else:
                self.logQueue.put("check")



        self.isWhiteTurn = not self.isWhiteTurn

        self.timeRulesTurnEnd()
        return True

    def parse(self, text):
        pattern = r'^[a-zA-Z]\d+,[a-zA-Z]\d+$'
        if not re.match(pattern, text):
            return None, None

        origin, new = text.split(",")

        originx, originy = origin[0], int(origin[1])
        originx = abc.index(originx.lower())
        newx, newy = new[0], int(new[1])
        newx = abc.index(newx.lower())

        origin = (originx,originy)
        new = (newx, newy)

        return origin, new


    def createChessboard(self):
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(0, 0, DIMS["tile"]*8, DIMS["tile"]*8)


        self.scene.mousePressEvent = self.canvasClicked
        self.scene.mouseReleaseEvent = self.canvasReleased
        self.scene.mouseMoveEvent = self.canvasDrag
        self.grayB = QBrush(Qt.gray)
        self.greenB = QBrush(QColor(0,255,0,128))
        self.whiteB = QBrush(Qt.white)
        self.pen = QPen(Qt.black)

        self.graphic = QGraphicsView(self.scene, self)
        self.graphic.setGeometry(0, 0, DIMS["tile"]*8, DIMS["tile"]*8)
        self.grid.addWidget(self.graphic,0,0, 10, 1)
        self.checkerPatter()


    def checkerPatter(self):
        for i in range(8):
            for j in range(8):
                if (i+j) % 2 == 0:
                    self.scene.addRect(i*DIMS["tile"], j*DIMS["tile"], DIMS["tile"], DIMS["tile"], self.pen, self.whiteB)
                else:
                    self.scene.addRect(i * DIMS["tile"],j * DIMS["tile"], DIMS["tile"], DIMS["tile"], self.pen, self.grayB)



    def showAvailableMoves(self, ):
        for move in self.availableMoves:
            x, y = move

            temp = self.scene.addRect(x * DIMS["tile"], y * DIMS["tile"], DIMS["tile"] , DIMS["tile"] , self.pen, self.greenB)
            self.availableObjects.append(temp)


    def hideAvailableMoves(self):
        for object in self.availableObjects:
            self.scene.removeItem(object)

        self.availableObjects = []

    def canvasDrag(self,event):
        if self.board.selectedPiece is None:
            return
        pos = event.scenePos()
        self.board.drag(pos)



    def canvasReleased(self,event):
        if self.board.selectedPiece is None:
            return

        new = self.getCoords(event)

        if self.board.selectedPiece.getPos() == new or new not in self.availableMoves:
            self.releasePiece()
            return


        x1,y1,newx,newy = self.board.move(self.board.selectedPiece.getPos(), new)

        self.releasePiece()

        self.history.saveMove(((x1,y1),(newx,newy)))
        self.logQueue.put("Moved from {x1},{y1} to {x2},{y2}.\n".format(x1=abc[x1],y1=str(y1),x2=abc[newx],y2=str(newy)))

        if self.board.checkCheck(self.board.map, not self.isWhiteTurn):
            if self.board.checkMate(not self.isWhiteTurn):
                self.logQueue.put("checkmate")
                self.gameFinished = True

            else:
                self.logQueue.put("check")

        self.isWhiteTurn = not self.isWhiteTurn

        self.timeRulesTurnEnd()
        self.boards.append(deepcopy(self.board.map))
        self.boardIndex = self.boardIndex + 1


        if self.config.mode == "PvP_Online":
            # if self.isServer and not self.isWhiteTurn:
            #     return
            # if not self.isServer and self.isWhiteTurn:
            #     return
            uniMove = self.history.convertToUniversalMove(((x1, y1), (newx, newy)))
            uniMove = "MOVE|" + uniMove
            self.client.send_text_message(uniMove)

            # if self.isServer:
            #     uniMove = self.history.convertToUniversalMove(((x1,y1),(newx,newy)))
            #     uniMove = "MOVE|" + uniMove
            #     self.client.send_text_message(uniMove)
            #     self.awaitMove()
            # else:
            #     uniMove = self.history.convertToUniversalMove(((x1,y1),(newx,newy)))
            #     uniMove = "MOVE|" + uniMove
            #     self.client.send_text_message(uniMove)
            #     self.awaitMoveClient()

        if self.robots and not self.gameFinished:
            self.randomMove()


    # def awaitMove(self):
    #     serverValidation = False
    #     while not serverValidation:
    #         move =self.moveQueue.get()
    #         origin, new = self.history.universalToPos(move)
    #         if self.execute(origin,new):
    #             serverValidation = True
    #
    #     pass
    #
    # def awaitMoveClient(self):
    #     move = self.moveQueue.get()
    #     print(move)
    #     origin, new = self.history.universalToPos(move)
    #     print(origin,new)
    #     self.execute(origin, new)

    def releasePiece(self):
        self.board.selectedPiece.deselect()
        self.board.selectedPiece = None
        self.hideAvailableMoves()


    def canvasClicked(self, event):
        if self.config.mode == "PvP_Online":
            if self.isServer and not self.isWhiteTurn:
                return
            if not self.isServer and self.isWhiteTurn:
                return

        x,y = self.getCoords(event)

        if self.board.isEmpty([x,y]) or self.selectedPiece is not None:
            return

        if not self.board.checkTurn([x,y], self.isWhiteTurn):
            return

        self.availableMoves = self.board.selectPiece(x,y)
        self.showAvailableMoves()

    def getCoords(self,event):
        pos = event.scenePos()
        x = int(pos.x() / DIMS["tile"])
        y = int(pos.y() / DIMS["tile"])
        return (x,y)
    
    


    def randomMove(self):

        bstate = self.board.encodeBoard()
        bstate = np.array(bstate)
        moves = self.board.getAllMovesEncoded(False)


        o, n = convNetwork.getBestMove(moves, bstate, net)


        self.board.move(o, n)

        self.hideAvailableMoves()
        if self.board.checkCheck(self.board.map, not self.isWhiteTurn):
            if self.board.checkMate(not self.isWhiteTurn):
                self.logQueue.put("checkmate")
                self.gameFinished = True

            else:
                self.logQueue.put("check")
        self.isWhiteTurn = not self.isWhiteTurn



class SetupWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.setWindowTitle('PyQt Chess')
        self.setGeometry(100, 100, 1000, 800)
        w = QWidget()
        self.setCentralWidget(w)
        self.grid = QGridLayout(w)


        self.modeGroup = QButtonGroup()
        self.timeGroup = QButtonGroup()


        # mode selection
        r1 = QRadioButton('PvP')
        r2 = QRadioButton('PvP_Online')
        r3 = QRadioButton('PvAI')
        self.grid.addWidget(r1, 1, 0)
        self.grid.addWidget(r2, 1 ,1)
        self.grid.addWidget(r3, 1, 2)
        self.modeGroup.addButton(r1)
        self.modeGroup.addButton(r2)
        self.modeGroup.addButton(r3)

        # time selection
        r1 = QRadioButton(gameTimes[0])
        r2 = QRadioButton(gameTimes[1])
        r3 = QRadioButton(gameTimes[2])
        self.grid.addWidget(r1, 2, 0)
        self.grid.addWidget(r2, 2 ,1)
        self.grid.addWidget(r3, 2, 2)
        self.timeGroup.addButton(r1)
        self.timeGroup.addButton(r2)
        self.timeGroup.addButton(r3)

        # ip settings
        self.ipField = QLineEdit()
        self.ipField.setPlaceholderText('Enter ip here')
        self.grid.addWidget(self.ipField,3,1)
        self.portField = QLineEdit()
        self.portField.setPlaceholderText('Enter port here')
        self.grid.addWidget(self.portField,3,2)

        # save/load
        self.button = QPushButton('Save Config')
        self.grid.addWidget(self.button, 4, 1)
        self.button.clicked.connect(self.save)

        self.button = QPushButton('Load Config')
        self.grid.addWidget(self.button, 4, 2)
        self.button.clicked.connect(self.load)

        # confirm
        self.button = QPushButton('Play')
        self.grid.addWidget(self.button,5,0)
        self.button.clicked.connect(self.play)


        self.show()


    def confirm(self):
        if self.modeGroup.checkedButton():
            self.config.mode = self.modeGroup.checkedButton().text()
        else:
            self.config.mode = "PvP"

        if self.timeGroup.checkedButton():
            self.config.time = self.timeGroup.checkedButton().text()
        else:
            self.config.time = "None"

        self.config.IP = self.ipField.text()
        self.config.port = int(self.portField.text())


    def save(self):
        self.confirm()
        self.config.save()

    def load(self):
        self.config.load()
        self.ipField.setText(self.config.IP)
        self.portField.setText(str(self.config.port))
        ChessWindow(self.config)
        self.close()


    def play(self):
        self.confirm()
        ChessWindow(self.config)
        self.close()




if __name__ == '__main__':

    app = QApplication(sys.argv)
    config = GameConfig()
    setup = SetupWindow(config)
    sys.exit(app.exec_())
