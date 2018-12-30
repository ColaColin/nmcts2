'''
Created on Apr 3, 2018

@author: cclausen
'''


import multiprocessing as mp
import importlib
import os
import time

import torch.optim as optim

from core.AbstractTorchLearner import AbstractTorchLearner
from core.NeuralMctsTrainer import NeuralMctsTrainer, PlayerComparator
from core.NeuralMctsPlayer import NeuralMctsPlayer

from core.misc import openJson

from nets.Nets import ResCNN, count_parameters

def object_for_class_name(module_name):
    class_name = module_name.split(".")[-1]
    m = importlib.import_module(".".join(module_name.split(".")[:-1]))
    c = getattr(m, class_name)
    return c()

def unrollLearningRates(lrs):
    result = []
    
    for ls in lrs:
        for _ in range(ls[0]):
            result.append(ls[1])
    
    return result

class ConfigLearner(AbstractTorchLearner):
    def __init__(self, config, gameInit):
        lconf = config["learning"]
        super(ConfigLearner, self).__init__(lconf["framesBufferSize"], 
                                            lconf["batchSize"],
                                            lconf["epochs"],
                                            unrollLearningRates(lconf["learningRates"]))
        self.config = config
        self.gameInit = gameInit
        self.initState(None)
        
    def clone(self):
        c = ConfigLearner(self.config, self.gameInit)
        
        if (self.net != None):
            c.initState(None)
            c.net.load_state_dict(self.net.state_dict())
            
        return c
    
    def getNetInputShape(self):
        dims = self.gameInit.getGameDimensions()
        return (dims[2], dims[0], dims[1])
    
    def getPlayerCount(self):
        return self.gameInit.getPlayerCount()
    
    def getMoveCount(self):
        return self.gameInit.getMoveCount()
    
    def createNetwork(self):
        netc = self.config["network"] 
        if (netc["type"] == "ResNet"):
            dims = self.gameInit.getGameDimensions()
            result = ResCNN(dims[0], dims[1], dims[2], 
                          netc["firstBlockKernelSize"], netc["firstBlockFeatures"], 
                          netc["blockFeatures"], netc["blocks"], self.getMoveCount(), 
                          self.getPlayerCount())
            
            print("Created a network with %i parameters" % count_parameters(result))
           
            return result
        else:
            assert False, "unsupported network type"

    def createOptimizer(self, net):
        if self.config["learning"]["useAdam"]:
            return optim.Adam(net.parameters(), weight_decay = 0.0001)
        else:
            return optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)

    def fillNetworkInput(self, state, tensor, batchIndex):
        self.gameInit.fillNetworkInput(state, tensor, batchIndex)

def ensureExists(d):
    if not os.path.exists(d):
        os.mkdir(d)

def pickComparativeConfig(config, key):
    config["network"] = config[key]["network"]
    config["learning"] = config[key]["learning"]

def createPlayerFor(config, initObject):
    learner = ConfigLearner(config, initObject)
    player = NeuralMctsPlayer(initObject.getStateTemplate(), config, learner)
    return player

def createTrainerFor(config, player, datadir, pool = None):
    lconf = config["learning"]
    trainer = NeuralMctsTrainer(player, datadir, lconf["framesPerIteration"], championGames = lconf["testGames"],
                                useTreeFrameGeneration = lconf["useTreeFrameGeneration"],
                                batchSize = lconf["batchSize"],
                                threads=lconf["threads"],
                                benchmarkTime=lconf["testInterval"],
                                pool = pool,
                                reAugmentEvery = lconf["reAugmentEvery"])
    return trainer

def runSingleTraining(workdir):
    datadir = os.path.join(workdir, "data")
    
    ensureExists(datadir)
   
    config = openJson(os.path.join(workdir, "config.json"))
    
    initObject = object_for_class_name(config["game"]["init"]) 
    initObject.setConfig(config)
    
    player = createPlayerFor(config, initObject)
    trainer = createTrainerFor(config, player, datadir)
    
    trainer.iterateLearning()

def runComparativeTraining(workdir):
    
    datadirA = os.path.join(workdir, "A")
    datadirB = os.path.join(workdir, "B")
    
    ensureExists(datadirA)
    ensureExists(datadirB)
    
    configA = openJson(os.path.join(workdir, "config.json"))
    configB = openJson(os.path.join(workdir, "config.json"))
    config = openJson(os.path.join(workdir, "config.json"))

    pool = mp.Pool(processes=config["compare"]["threads"])
    
    pickComparativeConfig(configA, "A")
    pickComparativeConfig(configB, "B")
    
    initA = object_for_class_name(configA["game"]["init"])
    initA.setConfig(configA)
    initB = object_for_class_name(configB["game"]["init"])
    initB.setConfig(configB)
    
    playerA = createPlayerFor(configA, initA)
    playerB = createPlayerFor(configB, initB)
    
    trainerA = createTrainerFor(configA, playerA, datadirA, pool = pool)
    trainerB = createTrainerFor(configB, playerB, datadirB, pool = pool)
    
    while True:
        print("Iterating on player A....")
        aTime = time.time()
        trainerA.iterateLearning(iterateForever = False)
        aTime = time.time() - aTime
        
        print("Iterating on player B....")
        bTime = time.time()
        trainerB.iterateLearning(iterateForever = False)
        bTime = time.time() - bTime
        
        print("Comparing player A and B")
        comp = PlayerComparator(config["compare"]["threads"], config["compare"]["testGames"], pool)
        aWins, bWins, draws, _ = comp.compare(playerA, playerB)

        print("A took %i, B took %i" % (aTime, bTime))
        print("A:B:Draws is %i : %i : %i" % (aWins, bWins, draws))

if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    #workdir = "/MegaKeks/nmcts2/c6_13_compare_8"
    #runComparativeTraining(workdir)

    workdir = "/MegaKeks/nmcts2/c6_19_test"
    runSingleTraining(workdir)
    
    