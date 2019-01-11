'''
Created on Jan 10, 2019

@author: cclausen
'''

import multiprocessing as mp
import os

from core.NeuralMctsTrainer import NeuralMctsTrainer
from core.NeuralMctsPlayer import NeuralMctsPlayer

from core.misc import openJson

from core.ConfigRunner import object_for_class_name, ConfigLearner

import subprocess

import time

def cloudictToMe(cMove):
    if cMove is None:
        return []
    
    cMove = cMove[5:]
    moves = []
    
    charsX = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    charsY = "SRQPONMLKJIH"
    
    for mi in range(len(cMove)//2):
        x = charsX.find(cMove[mi*2])
        y = charsY.find(cMove[mi*2+1])
        print("Cloudict to me yields ", x, y)
        key = y * 12 + x
        moves.append(key)
    
    return moves

def meToCloudict(meMove):
    charsX = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    charsY = "SRQPONMLKJIH"
    
    result = "move "
    
    for key in meMove:
        mx = key % 12
        my = key // 12
        print("my move position is ", mx, my)
        result += charsX[int(mx)]
        result += charsY[int(my)]
    
    return result

def playVsCloudict(exePath, searchDepth, cloudictMovesFirst, playFunc):
    proc = subprocess.Popen([exePath], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    
    proc.stdin.write(("depth " + str(searchDepth) + "\n").encode())
    proc.stdin.flush()
    
    cloudictMove = None
    
    if cloudictMovesFirst:
        proc.stdin.write("new black\n".encode())
        proc.stdin.flush();
        cloudictMove = proc.stdout.readline().decode()
    else:
        proc.stdin.write("new white\n".encode())
        proc.stdin.flush();

    playMove = None
    
    gameWinner = -1
    
    while True:
        if cloudictMove is not None:
            print("Got cloudict move: " + cloudictMove)
        
        thinkStart = time.time()
        
        gameWinner, playMove = playFunc(cloudictToMe(cloudictMove))
    
        print("My thinking time was %f sec" % (time.time() - thinkStart))
    
        if playMove is None:
            break
    
        myMove = meToCloudict(playMove)
    
        print("My move for cloudict is: " + myMove)
        
        thinkStart = time.time()
        
        proc.stdin.write((myMove+"\n").encode())
        proc.stdin.flush()
        cloudictMove = proc.stdout.readline().decode()
        
        print("Cloudict thinking time was %f sec" % (time.time() - thinkStart))
    
    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout=0.2)
    
    if cloudictMovesFirst:
        return gameWinner == 0
    else:
        return gameWinner == 1
    
def pickComparativeConfig(config, key):
    config["network"] = config[key]["network"]
    config["learning"] = config[key]["learning"]

if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    workdir = "/ImbaKeks/nmcts/newDrawOutput"
    playVersion = None #"B"
    firstMove = 0
    
    if playVersion == None:
        datadir = os.path.join(workdir, "data")
    else:
        datadir = os.path.join(workdir, playVersion)

    config = openJson(os.path.join(workdir, "config.json"))
    
    if playVersion != None:
        pickComparativeConfig(config, playVersion)
    
    initObject = object_for_class_name(config["game"]["init"])
    
    initObject.setConfig(config)
    
    learner = ConfigLearner(config, initObject)
    player = NeuralMctsPlayer(initObject.getStateTemplate(), config, learner)
    lconf = config["learning"]
    trainer = NeuralMctsTrainer(player, datadir, lconf["framesPerIteration"], championGames = lconf["testGames"],
                                useTreeFrameGeneration = lconf["useTreeFrameGeneration"],
                                batchSize = lconf["batchSize"],
                                threads=lconf["threads"],
                                benchmarkTime=lconf["testInterval"])
    
    trainer.load(loadFrames = False)
    
    playFunc = trainer.learner.getPlayFunc()
    
    playVsCloudict("/ImbaKeks/cloudict/gameEngine", 7, True, playFunc)
    
    