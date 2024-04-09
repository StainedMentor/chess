import json

BOARD_DEF=[["rd","nd","bd","qd","kd","bd","nd","rd"],
         ["pd","pd","pd","pd","pd","pd","pd","pd"],
         [None,None,None,None,None,None,None,None],
         [None,None,None,None,None,None,None,None],
         [None,None,None,None,None,None,None,None],
         [None,None,None,None,None,None,None,None],
         ["pl","pl","pl","pl","pl","pl","pl","pl"],
         ["rl","nl","bl","ql","kl","bl","nl","rl"]]

DIMS = {"tile":75}

abc = ["a","b","c","d","e","f","g","h"]

numabc = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7}

gameTimes = ["None", "Bullet", "Blitz", "Increment"]

gameModes = ["PvP", "PvP_Online", "PvAI"]



class GameConfig:
    def __init__(self, mode=gameModes[0], time=gameTimes[0],ip = "", port = 0):
        self.mode = mode
        self.timeSetting = time
        self.IP = ip
        self.port = port

    def save(self):
        dict = self.__dict__

        jsonString = json.dumps(dict)

        with open('config.json', 'w') as f:
            f.write(jsonString)

    def load(self):
        with open('config.json', 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            setattr(self, key, value)