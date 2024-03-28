import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGraphicsView, \
    QGraphicsScene, QGridLayout, QTextEdit, QScrollArea, QLabel, QLineEdit
from PyQt5.QtGui import QBrush, QPen, QColor
from PyQt5.QtCore import Qt,QTimer

import sys
import queue
import random
import re

from constants import *
from board import Board

from network import ChessNet
net = ChessNet()

net.load_state_dict(torch.load('dict.pth'))
net.eval()

class ChessWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQt Chess')
        self.setGeometry(100, 100, 1000, 800)

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



        self.addUI()
        self.createChessboard()



        self.board = Board(self.scene)
        self.board.initPieces()


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.handleLog)
        self.timer.start(1000)



        self.show()

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
        self.robotButton = QPushButton("Toggle robots: " + str(self.robots), w)
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

        self.timerToggle = QPushButton("Toggle timer: " + gameTimes[self.timeType], w)
        self.timerToggle.clicked.connect(self.toggleTime)
        self.grid.addWidget(self.timerToggle, 8, 1)

        # input
        self.command = QLineEdit(self)
        self.command.setPlaceholderText('A0,A1')
        self.submit_command = QPushButton('Submit', self)
        self.submit_command.clicked.connect(self.execute)
        self.grid.addWidget(self.command, 4, 1)
        self.grid.addWidget(self.submit_command, 5, 1)

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
        self.robots = not self.robots
        self.robotButton.setText("Toggle robots: " + str(self.robots))

    def execute(self):
        text = self.command.text()
        origin, new = self.parse(text)
        if origin is None:
            self.logQueue.put("invalid format\n")
            return
        piece = self.board.map[origin[1]][origin[0]]
        if piece is None:
            self.logQueue.put("invalid move (no piece)\n")
            return
        self.availableMoves = piece.getAvailableMoves(self.board.map)

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
            else:
                self.logQueue.put("check")

        self.isWhiteTurn = not self.isWhiteTurn

        self.timeRulesTurnEnd()

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


        self.scene.mousePressEvent = self.canvasClicked
        self.scene.mouseReleaseEvent = self.canvasReleased
        self.scene.mouseMoveEvent = self.canvasDrag
        self.grayB = QBrush(Qt.gray)
        self.greenB = QBrush(QColor(0,255,0,128))
        self.whiteB = QBrush(Qt.white)
        self.pen = QPen(Qt.black)

        self.graphic = QGraphicsView(self.scene, self)
        self.graphic.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphic.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
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

        self.logQueue.put("Moved from {x1},{y1} to {x2},{y2}.\n".format(x1=abc[x1],y1=str(y1),x2=abc[newx],y2=str(newy)))

        if self.board.checkCheck(self.board.map, not self.isWhiteTurn):
            if self.board.checkMate(not self.isWhiteTurn):
                self.logQueue.put("checkmate")
            else:
                self.logQueue.put("check")

        self.isWhiteTurn = not self.isWhiteTurn

        self.timeRulesTurnEnd()

        if self.robots:
            self.randomMove()


    def releasePiece(self):
        self.board.selectedPiece.deselect()
        self.board.selectedPiece = None
        self.hideAvailableMoves()


    def canvasClicked(self, event):
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
        # print(bstate)

        bstate = bstate.flatten()
        moves = self.board.getAllMovesEncoded(False)

        vals = torch.Tensor()
        for move in moves:
            o, n = move
            inVec = np.concatenate((bstate, o, n))
            inVec = torch.Tensor(inVec)
            output = net(inVec)
            vals = torch.cat((vals, output))

        vals = vals.detach().numpy()
        bestIndex = vals.argmax()

        o, n = moves[bestIndex]

        x1, y1 = o
        x, y = n

        if self.board.map[y][x] is not None:
            self.board.map[y][x].delete(self.scene)

        self.board.map[y][x] = self.board.map[y1][x1]
        self.board.map[y1][x1] = None
        self.board.map[y][x].move(x, y)
        self.selectedPiece = None
        self.hideAvailableMoves()
        self.isWhiteTurn = not self.isWhiteTurn


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChessWindow()
    sys.exit(app.exec_())
