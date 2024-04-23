import re
import sqlite3
import xml.etree.ElementTree as ET

from constants import *
from copy import deepcopy
class History:

    def __init__(self):

        self.board = deepcopy(BOARD_DEF)
        self.moveArray = []

        self.initSql()
        self.initXML()

    def initSql(self):
        self.sql = sqlite3.connect('chess.db')
        self.sqlCursor = self.sql.cursor()
        self.sqlCursor.execute('''CREATE TABLE IF NOT EXISTS moves (
                            id INTEGER PRIMARY KEY,
                            move TEXT NOT NULL
                        )''')

    def initXML(self):
        self.XMLroot = ET.Element("moves")
        self.tree = ET.ElementTree(self.XMLroot)
        with open('chess.xml', 'wb') as f:
            self.tree.write(f)

    def saveMove(self, move):
        self.moveArray.append(move)

        universalMove = self.convertToUniversalMove(move)

        self.saveToSQL(universalMove)
        self.saveToXML(universalMove)


    def saveToSQL(self, move):
        self.sqlCursor.execute('INSERT INTO moves (move) VALUES (?)', (move,))
        self.sql.commit()


    def saveToXML(self,move):
        move_element = ET.SubElement(self.XMLroot, "move")
        move_element.text = move
        with open('chess.xml', 'wb') as f:
            self.tree.write(f)

    def convertToUniversalMove(self, move):
        origin, new = move
        originx, originy = origin
        newx, newy= new
        newx = abc[newx]
        originx = abc[originx]
        newy = str(newy)
        originy = str(originy)

        return originx + originy + "," + newx + newy

    def universalToPos(self, text):
        pattern = r'^[a-zA-Z]\d+,[a-zA-Z]\d+$'
        if not re.match(pattern, text):
            return None, None

        origin, new = text.split(",")

        originx, originy = origin[0], int(origin[1])
        originx = abc.index(originx.lower())
        newx, newy = new[0], int(new[1])
        newx = abc.index(newx.lower())

        origin = (originx, originy)
        new = (newx, newy)

        return origin, new

