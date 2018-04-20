'''
Created on Apr 4, 2018

@author: cclausen
'''

import multiprocessing as mp
import os

from core.NeuralMctsTrainer import NeuralMctsTrainer
from core.NeuralMctsPlayer import NeuralMctsPlayer

from core.misc import openJson

from core.ConfigRunner import object_for_class_name, ConfigLearner

def pickComparativeConfig(config, key):
    config["network"] = config[key]["network"]
    config["learning"] = config[key]["learning"]

if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    workdir = "/MegaKeks/nmcts2/c6_13_speed_test_2"
    playVersion = None #"B"
    firstMove = 1
    
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
    
    trainer.load()
    trainer.learner.playVsHuman(initObject.getStateTemplate(), firstMove, [], initObject.stateFormat, initObject.mkParseCommand())