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

if __name__ == '__main__':
    mp.set_start_method("spawn")
    
    #workdir = sys.argv[1]
    workdir = "/MegaKeks/nmcts2/mnk333_A"
    firstMove = 1
    
    
    
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
    
    trainer.load()
    trainer.learner.playVsHuman(initObject.getStateTemplate(), firstMove, [], initObject.stateFormat, initObject.mkParseCommand())