import random
from copy import deepcopy

import numpy as np

from ai.boardNoGui import Board

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import time


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*8*8*2,128)
        self.fc2 = nn.Linear(128,1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.flatten()
        # x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


def getBestMove(moves, bstate, net):
    vals = torch.Tensor()
    for move in moves:
        inVec = readyInput(bstate,move)
        output = net(inVec)
        vals = torch.cat((vals, output))
    vals = vals.detach().numpy()
    bestIndex = vals.argmax()
    return moves[bestIndex]

def updateRewards(buffer, num, modifier, fade,offset=0):
    if num == -1:
        num = int(len(buffer)/2)

    for i in range(offset,num*2,2):
        if i >= len(buffer): break
        state, action, reward = buffer[-1 - i]
        newReward = reward + (modifier/(fade*i+1))

        buffer[-1 - i] = [state, action, newReward]

def decodeBoard(boardState):
    temp = np.array(boardState)
    temp = temp.reshape(8,8)
    return temp

def readyInput(state,move):
    temp = deepcopy(state)
    s = deepcopy(state)
    o, n = move
    x1,y1 = o
    x2,y2 = n
    temp[y2][x2] = temp[y1][x1]
    temp[y1][x1] = 0
    full = np.stack((s,temp))
    return torch.Tensor(full)



def train():
    # init
    start = time.time()
    net = ChessNet()
    net.load_state_dict(torch.load('dict_conv.pth'))

    num_episodes = 1000
    maxSteps = 120
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    memoryBuffer = []
    stats = None

    gameSamples = []
    lossArr = []

    # Training loop
    for episode in range(num_episodes):
        whiteTook, blackTook, whiteWon, blackWon = 0,0,0,0
        newBoard = Board()
        newBoard.initPieces()

        episodeBuffer = []
        episodeEnded = False
        for step in range(maxSteps):

            turn = (step+1)%2
            bstate = newBoard.encodeBoard()
            bstate = np.array(bstate)
            moves = newBoard.getAllMovesEncoded(True if turn == 1 else False)


            o, n = getBestMove(moves,bstate,net)

            newBoard.move(o, n)
            check = newBoard.checkCheck(newBoard.map, not (True if turn == 1 else False))
            mate = newBoard.checkMate(not (True if turn == 1 else False))

            reward = 0
            if check:
                reward = 5

            if mate:
                episodeEnded = True


            newState = newBoard.encodeBoard()
            newState = np.array(newState)
            newState = newState.flatten()

            p1 = np.count_nonzero(bstate)
            p2 = np.count_nonzero(newState)
            if p2 < p1:
                reward += 10
                updateRewards(episodeBuffer,3,10,0.1,1)
                updateRewards(episodeBuffer,3,-10,0.1)

                if turn == 1:
                    whiteTook += 1
                else:
                    blackTook += 1

            episodeBuffer.append([bstate, (o, n), reward])

            if episodeEnded:
                if turn == 1:
                    whiteWon += 1
                else:
                    blackWon += 1
                updateRewards(episodeBuffer,-1,100,0)
                updateRewards(episodeBuffer,-1,-100,0.1,1)

                break
        current = np.array([whiteTook,blackTook,whiteWon,blackWon])
        if stats is None:
            stats = np.array([current])
        else:
            stats = np.append(stats,[current], axis=0)
        print("\r"+"episode: " + str(episode+1)+ " white: " + str(np.sum(stats[:,2]))+ " black: " + str(np.sum(stats[:,3])) + " finished " + str((np.sum(stats[:,2]) +np.sum(stats[:,3]))/(episode+1)) + " minutes:" + str(round((time.time()-start)/60)), end="")

        # if not episodeEnded:
        #     updateRewards(episodeBuffer,-1,-40,0)
        #     updateRewards(episodeBuffer,-1,-40,0,1)

        memoryBuffer.extend(episodeBuffer)


        # Network training
        batchSize = min(len(memoryBuffer), 64)
        miniBatch = random.sample(memoryBuffer, batchSize)
        batchSizeEpisode = min(len(episodeBuffer), 64)
        miniBatchEpisode = random.sample(episodeBuffer, batchSizeEpisode)


        miniBatch.extend(episodeBuffer)
        # miniBatch = episodeBuffer

        tempLoss = 0
        for state, action, reward in miniBatch:
            inVec = readyInput(state,action)
            predicted_values = net(inVec)

            target = [reward]
            target = torch.Tensor(target)
            loss = F.l1_loss(predicted_values, target)

            tempLoss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lossArr.append(loss/len(miniBatch))


        if episode % 10 ==0:
            gameSamples.append(episodeBuffer)


    # for sample in gameSamples:
    #     for turn in sample:
    #         print(decodeBoard(turn[0]))


    stDict = net.state_dict()
    torch.save(stDict, 'dict_conv.pth')

    stats = np.array(stats)

    print("")
    print(np.sum(stats[:,2]),np.sum(stats[:,3]))

    plt.plot(np.arange(0,num_episodes), stats[:,0], marker='o', linestyle='-')
    plt.plot(np.arange(0,num_episodes), stats[:,1], marker='o', linestyle='-')

    plt.xlabel('Iteration')
    plt.ylabel('Pieces taken')

    plt.grid(True)
    plt.show()

    lossArr = torch.Tensor(lossArr)
    lossArr = lossArr.detach().numpy()

    plt.plot(np.arange(0,num_episodes), lossArr, marker='o', linestyle='-')

    plt.xlabel('Iteration')
    plt.ylabel('Pieces taken')

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train()