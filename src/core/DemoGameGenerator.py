'''
Created on Jan 8, 2019

@author: cclausen
'''
import multiprocessing as mp
import os

from core.NeuralMctsTrainer import NeuralMctsTrainer
from core.NeuralMctsPlayer import NeuralMctsPlayer

from core.misc import openJson

from core.ConfigRunner import object_for_class_name, ConfigLearner

import json

def writeJson(path, data, pretty = True):
    with open(path, "w") as f:
        f.truncate()
        if pretty:
            f.write(json.dumps(data, sort_keys=True, indent=4))
        else:
            f.write(json.dumps(data))


def pickComparativeConfig(config, key):
    config["network"] = config[key]["network"]
    config["learning"] = config[key]["learning"]

if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    iteration = None
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
    
    trainer.load(loadFrames = False, iteration=iteration)
    package = trainer.learner.recordDemoGame()
    
    writeJson(os.path.join(workdir, "gamerecord.json"), package, pretty=False)