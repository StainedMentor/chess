import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QGraphicsView, \
    QGraphicsScene, QDesktopWidget, QGraphicsPixmapItem, QGridLayout, QTextEdit, QScrollArea, QDialog
from PyQt5.QtGui import QBrush, QPainter, QPen, QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer
from Pieces import *
from board import Board
import queue

import random



class ChessWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.selectedPiece = None
        self.availableObjects = []
        self.availableMoves = []
        self.isWhiteTurn = True
        self.robots = False
        self.log = ""
        self.logQueue = queue.Queue()
        self.log_field = QTextEdit()

        self.setWindowTitle('PyQt Chess')
        self.setGeometry(100, 100, 800, 500)

        self.addUI()
        self.createChessboard()



        self.board = Board(self.scene)
        self.board.initPieces()


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.handleLog)
        self.timer.start(500)

        self.show()

    def handleLog(self):
        while not self.logQueue.empty():
            self.log += self.logQueue.get()
            self.log_field.setPlainText(self.log)

    def addUI(self):
        w = QWidget()
        self.setCentralWidget(w)
        grid = QGridLayout(w)
        self.robotButton = QPushButton("Toggle robots: " + str(self.robots), w)
        self.robotButton.clicked.connect(self.robotsStart)

        self.setCentralWidget(w)
        grid.addWidget(self.robotButton, 1, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)

        self.log_field.setPlainText(self.log)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.log_field)
        grid.addWidget(scroll_area, 0, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)


    def robotsStart(self):
        self.robots = not self.robots
        self.robotButton.setText("Toggle robots: " + str(self.robots))


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
        if self.selectedPiece is None:
            return
        x, y = event.scenePos().x(), event.scenePos().y()
        self.selectedPiece.setPos(x,y)


    def canvasReleased(self,event):
        if self.selectedPiece is None:
            return

        new = self.getCoords(event)

        if self.selectedPiece.getPos() == new or new not in self.availableMoves:
            self.releasePiece()
            return


        self.board.move(self.selectedPiece.getPos(), new)

        self.releasePiece()



        self.isWhiteTurn = not self.isWhiteTurn
        # self.logQueue.put("Moved from {x1},{y1} to {x2},{y2}.\n".format(x1=str(x1),y1=str(y1),x2=str(newx),y2=str(newy)))

        if self.robots:
            self.randomMove()


    def releasePiece(self):
        self.selectedPiece.deselect()
        self.selectedPiece = None
        self.hideAvailableMoves()


    def canvasClicked(self, event):
        x,y = self.getCoords(event)

        if self.board.isEmpty([x,y]) or self.selectedPiece is not None:
            return

        if not self.board.checkTurn([x,y], self.isWhiteTurn):
            return

        self.selectedPiece = self.board.map[y][x]
        self.selectedPiece.select()
        self.availableMoves = self.selectedPiece.getAvailableMoves(self.board.map)
        self.showAvailableMoves()

    def getCoords(self,event):
        pos = event.scenePos()
        x = int(pos.x() / DIMS["tile"])
        y = int(pos.y() / DIMS["tile"])
        return (x,y)
    
    
    # random bot
    def getAllMoves(self):
        possibleMoves = []

        for x in range(8):
            for y in range(8):
                if self.pieceMap[y][x] is None:
                    continue
                if not self.pieceMap[y][x].isWhite:
                    available = self.pieceMap[y][x].getAvailableMoves(self.pieceMap)
                    for a in available:
                        possibleMoves.append(((x,y),a))
        return possibleMoves

    def randomMove(self):
        moves = self.getAllMoves()
        random_move = random.choice(moves)
        a, b = random_move
        x1, y1 = a
        x, y = b

        if self.pieceMap[y][x] is not None:
            self.pieceMap[y][x].delete(self.scene)

        self.pieceMap[y][x] = self.pieceMap[y1][x1]
        self.pieceMap[y1][x1] = None
        self.pieceMap[y][x].move(x, y)
        self.selectedPiece = None
        self.hideAvailableMoves()
        self.isWhiteTurn = not self.isWhiteTurn


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChessWindow()
    sys.exit(app.exec_())
