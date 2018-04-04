'''
Created on Apr 3, 2018

@author: cclausen
'''


import multiprocessing as mp
import importlib
import os

import torch.optim as optim

from core.AbstractTorchLearner import AbstractTorchLearner
from core.NeuralMctsTrainer import NeuralMctsTrainer
from core.NeuralMctsPlayer import NeuralMctsPlayer

from core.misc import openJson

from nets.Nets import ResCNN

def object_for_class_name(module_name):
    class_name = module_name.split(".")[1]
    m = importlib.import_module(module_name)
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
            return ResCNN(dims[0], dims[1], dims[2], 
                          netc["firstBlockKernelSize"], netc["firstBlockFeatures"], 
                          netc["blockFeatures"], netc["blocks"], self.getMoveCount(), 
                          self.getPlayerCount())
        else:
            assert False, "unsupported network type"

    def createOptimizer(self, net):
        # TOOD make this configurable!!!
        return optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)

    def fillNetworkInput(self, state, tensor, batchIndex):
        self.gameInit.fillNetworkInput(state, tensor, batchIndex)

if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    #workdir = sys.argv[1]
    
    workdir = "/MegaKeks/nmcts2/mnk333_A"
    
    datadir = os.path.join(workdir, "data")
    
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    
    config = openJson(os.path.join(workdir, "config.json"))
    
    initObject = object_for_class_name(config["game"]["init"]) 
    
    initObject.setConfig(config)
    
    learner = ConfigLearner(config, initObject)
    player = NeuralMctsPlayer(initObject.getStateTemplate(), config["learning"]["mctsExpansions"], learner)
    lconf = config["learning"]
    trainer = NeuralMctsTrainer(player, datadir, lconf["framesPerIteration"], championGames = lconf["testGames"],
                                useTreeFrameGeneration = lconf["useTreeFrameGeneration"],
                                batchSize = lconf["batchSize"],
                                threads=lconf["threads"],
                                benchmarkTime=lconf["testInterval"])
    
    trainer.iterateLearning()
    