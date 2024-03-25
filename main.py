import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QGraphicsView, \
    QGraphicsScene, QDesktopWidget, QGraphicsPixmapItem, QGridLayout, QTextEdit, QScrollArea, QDialog
from PyQt5.QtGui import QBrush, QPainter, QPen, QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer
from Pieces import *
import queue

import random


class PopupWindow(QDialog):
    def __init__(self, log, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Popup Window")

        layout = QVBoxLayout()
        label = QLabel(log)
        layout.addWidget(label)
        button = QPushButton("Close")
        button.clicked.connect(self.close)
        layout.addWidget(button)

        self.setLayout(layout)

class ChessWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('PyQt Chess')
        self.setGeometry(100, 100, 800, 500)
        self.pieceMap = BOARD_DEF
        self.selectedPiece = None
        self.availableObjects = []
        self.availableMoves = []
        self.isWhiteTurn = True
        self.robots = False
        self.log = ""
        self.logQueue = queue.Queue()
        self.log_field = QTextEdit()
        self.addUI()
        self.createChessboard()

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
        popup = PopupWindow(self.log)
        popup.exec_()

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
        self.graphic.setGeometry(0, 0, 400, 400)
        self.checkerPatter()
        self.addPieces()


    def checkerPatter(self):
        for i in range(8):
            for j in range(8):
                if (i+j) % 2 == 0:
                    self.scene.addRect(i*50, j*50, 50, 50, self.pen, self.whiteB)
                else:
                    self.scene.addRect(i * 50,j * 50, 50, 50, self.pen, self.grayB)

    def addPieces(self):
        for i in range(8):
            for j in range(8):
                if BOARD_DEF[j][i] is not None:
                    a = Piece(BOARD_DEF[j][i],i,j)
                    a.drawSelf(self.scene)

                    self.pieceMap[j][i] = a

    def showAvailableMoves(self, ):
        for move in self.availableMoves:
            x, y = move

            temp = self.scene.addRect(x * 50, y * 50, 50 , 50 , self.pen, self.greenB)
            self.availableObjects.append(temp)


    def hideAvailableMoves(self):
        for object in self.availableObjects:
            self.scene.removeItem(object)

        self.availableObjects = []

    def canvasDrag(self,event):
        if self.selectedPiece is None:
            return
        pos = event.scenePos()
        x1, y1 = self.selectedPiece
        self.pieceMap[y1][x1].setPos(pos.x(),pos.y())

        delta = event.pos() - self.last_pos

        self.graphic.horizontalScrollBar().setValue(int(self.graphic.horizontalScrollBar().value() - delta.x()))
        self.graphic.verticalScrollBar().setValue(int(self.graphic.verticalScrollBar().value() - delta.y()))
        self.last_pos = event.pos()


    def canvasReleased(self,event):
        if self.selectedPiece is None:
            return

        pos = event.scenePos()
        x = int(pos.x()/50)
        y = int(pos.y()/50)
        x1, y1 = self.selectedPiece

        if x == x1 and y == y1:
            self.selectedPiece = None
            self.pieceMap[y1][x1].deselect()
            self.hideAvailableMoves()

            return

        if (x, y) not in self.availableMoves:
            return

        if self.pieceMap[y][x] is not None:
            self.pieceMap[y][x].delete(self.scene)

        if self.pieceMap[y1][x1].type[0] == "k" and abs(x-x1) == 2:
            print(1)
            diff = x-x1
            if diff > 0:
                self.pieceMap[y][5] = self.pieceMap[y1][7]
                self.pieceMap[y][7] = None
                self.pieceMap[y][5].move(5, y)

            else:
                self.pieceMap[y][3] = self.pieceMap[y1][0]
                self.pieceMap[y][0] = None
                self.pieceMap[y][3].move(3, y)

        self.pieceMap[y][x] = self.pieceMap[y1][x1]
        self.pieceMap[y1][x1] = None
        self.pieceMap[y][x].move(x, y)
        self.selectedPiece = None
        self.hideAvailableMoves()

        self.isWhiteTurn = not self.isWhiteTurn
        self.logQueue.put("Moved from {x1},{y1} to {x2},{y2}.\n".format(x1=str(x1),y1=str(y1),x2=str(x),y2=str(y)))

        if self.robots:
            self.randomMove()

    def randomMove(self):
        moves = self.getAllMoves()
        random_move = random.choice(moves)
        a, b = random_move
        x1,y1= a
        x, y =  b

        if self.pieceMap[y][x] is not None:
            self.pieceMap[y][x].delete(self.scene)

        self.pieceMap[y][x] = self.pieceMap[y1][x1]
        self.pieceMap[y1][x1] = None
        self.pieceMap[y][x].move(x, y)
        self.selectedPiece = None
        self.hideAvailableMoves()
        self.isWhiteTurn = not self.isWhiteTurn


    def canvasClicked(self, event):
        self.last_pos = event.pos()

        pos = event.scenePos()
        x = int(pos.x()/50)
        y = int(pos.y()/50)

        if self.pieceMap[y][x] is None or self.selectedPiece is not None:
            return

        if self.pieceMap[y][x].isWhite == self.isWhiteTurn:

            self.selectedPiece = (x,y)
            self.pieceMap[y][x].select()
            self.availableMoves = self.pieceMap[y][x].getAvailableMoves(self.pieceMap)
            self.showAvailableMoves()



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



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChessWindow()
    sys.exit(app.exec_())
