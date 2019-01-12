'''

A class that takes an AbstractLearner and teaches it to play better using
expert iteration. The actual playing is done by NeuralMctsPlayer

Created on Oct 27, 2017

@author: cclausen
'''

import multiprocessing as mp
#import torch.multiprocessing as mp
import time
import random
import os
import pickle

import sys

from core.misc import openJson, writeJson

import numpy as np

# given an N player game:
# assert N % 2 == 0 #because I don't want to think about the more general case...
# setup N/2 players playing as the learner, N/2 as bestPlayer
# sum up all wins of the learner instances and all wins of the bestPlayer instances
class PlayerComparator():
    def __init__(self, threads, testGames, pool):
        self.pool = pool
        self.threads = threads
        self.testGames = testGames
        
    def compare(self, playerA, playerB):
        gamesPerProc = int(self.testGames / self.threads)
        
        playersN = playerA.stateTemplate.getPlayerCount()
        aPlayers = int(playersN / 2)
        bPlayers = int(playersN / 2)
        
        assert gamesPerProc % 2 == 0, "testgames / threads / 2 needs to be even!"
        
        missing = self.testGames % self.threads
        assert missing == 0, str(missing) + " != " + 0
        assert playersN % 2 == 0, "an uneven player count would create more issues and need a bit of code improvement in learnerIsNewChampion..."
        
        dbgFrames = False
        
        asyncs = []
        asyncsInverted = []
        for _ in range(self.threads):
            g = int(gamesPerProc / 2)
            asyncs.append(self.pool.apply_async(playerA.playAgainst, 
                args=(g, g, [playerA] * (aPlayers - 1) + [playerB] * bPlayers, dbgFrames)))
            asyncsInverted.append(self.pool.apply_async(playerB.playAgainst, 
                args=(g, g, [playerB] * (bPlayers - 1) + [playerA] * aPlayers, dbgFrames)))
        
        sumResults = [0,0,0]
        
        firstWins = 0
        
        for asy in asyncs:
            r, gframes = asy.get()
            
            if dbgFrames:
                for f in gframes[0]:
                    print(f[0].c6)
                    print(list(reversed(sorted(f[1])))[:5], f[1])
                    print(f[3])
                    print("...")
                
            sumResults[2] += r[-1]
            firstWins += r[0]
            for i in range(len(r)-1):
                if i < aPlayers:
                    sumResults[0] += r[i]
                else:
                    sumResults[1] += r[i]
        
        for asy in asyncsInverted:
            r, gframes = asy.get()
            
            if dbgFrames:
                for f in gframes[0]:
                    print(f[0].c6)
                    print(list(reversed(sorted(f[1])))[:5], f[1])
                    print(f[3])
                    print("...")
            
            sumResults[2] += r[-1]
            firstWins += r[0]
            for i in range(len(r)-1):
                if i < bPlayers:
                    sumResults[1] += r[i]
                else:
                    sumResults[0] += r[i]
        
        assert sum(sumResults) == self.testGames
        
        aWins = sumResults[0]
        bWins = sumResults[1]
        draws = sumResults[-1]
        
        return aWins, bWins, draws, firstWins
        
        
class NeuralMctsTrainer():
    
    def __init__(self, nplayer, workingdir, framesPerIteration, 
                 pool=None, useTreeFrameGeneration = False, 
                 championGames=500, batchSize=200, 
                 threads=5, benchmarkTime = 3600,
                 reAugmentEvery=1):
        self.learner = nplayer
        
        self.reAugmentEvery = reAugmentEvery
        
        self.bestIteration = 0
        
        self.useTreeFrameGeneration = useTreeFrameGeneration
        
        self.framesPerIteration = framesPerIteration
        
        self.lastBenchmarkTime = time.time()
        self.workingdir = workingdir
        if pool == None:
            self.pool = mp.Pool(processes=threads)
        else:
            self.pool = pool
        self.threads = threads
        self.batchSize = batchSize
        self.frameHistory = []
        self.championGames = championGames
        self.benchmarkTime = benchmarkTime
        
        gamesPerProc = int(self.championGames / self.threads)
        assert gamesPerProc % 2 == 0, "championgames / threads / 2 needs to be even!"
        
        missing = self.championGames % self.threads
        assert missing == 0, str(missing) + " != " + 0
        
        playersN = self.learner.stateTemplate.getPlayerCount()
        assert playersN % 2 == 0, "an uneven player count would create more issues and need a bit of code improvement in learnerIsNewChampion..."
        
        self.byThreadTreeGenerators = [None] * self.threads
    
    def benchmarkShowsProgress(self):
        self.bestPlayer = self.learner.clone();
        self.bestPlayer.learner.initState(os.path.join(self.workingdir, "learner.iter" + str(max(1, self.bestIteration))))
        
        comp = PlayerComparator(self.threads, self.championGames, self.pool)
        myWins, otherWins, draws, firstWins = comp.compare(self.learner, self.bestPlayer)
        
        eps = 0.00000001
        print("Learner wins %i, best player wins %i, %i draws, %i first move wins: Winrate of %f" % 
              (myWins, otherWins, draws, firstWins, (myWins + eps) / (myWins + otherWins + eps)))
        
        improved = myWins > (myWins + otherWins) * 0.55
        if improved:
            print("Progress was made")
            
        return improved
    
    def averageFrames(self, frameList):
        result = frameList[0]
        
        for i in range(1, len(frameList)):
            result[1] += frameList[i][1]
            result[2] += frameList[i][2]
            result[3] += frameList[i][3]
        
        result[1] /= len(frameList)
        result[2] /= len(frameList)
        result[3] /= len(frameList)

        return result
    
    def updateFrame(self, targetFrame, newFrame):
        # move probabilities, mostly use the new ones
        targetFrame[1] = 0.9 * newFrame[1] + 0.1 * targetFrame[1]
        
        # no idea what best value is even for...
        targetFrame[2] = 0.5 * newFrame[2] + 0.5 * newFrame[2]
        
        # winning probabilities, be less sure about changes
        targetFrame[3] = 0.5 * newFrame[3] + 0.5 * newFrame[3]
    
    def addNewFramesToHistory(self, newFrames, iteration):
        uniqueFrames = {}
        
        for newFrame in newFrames:
            if not newFrame[0] in uniqueFrames:
                uniqueFrames[newFrame[0]] = []
            uniqueFrames[newFrame[0]].append(newFrame)
        
        uniqList = []
        
        for key in uniqueFrames:
            uniqList.append(self.averageFrames(uniqueFrames[key]))
        
        print("Got %i unique frames!" % len(uniqList))
        
        knownFramesDict = {}
        for hidx, hframe in enumerate(self.frameHistory):
            knownFramesDict[hframe[0][0]] = hidx
        
        newFrames = 0
        
        for newFrame in uniqList:
            if newFrame[0] in knownFramesDict:
                frameIdx = knownFramesDict[newFrame[0]]
                self.updateFrame(self.frameHistory[frameIdx][0], newFrame)
                self.frameHistory[frameIdx][1] = time.time()
            else:
                newFrames += 1
                self.frameHistory.append([newFrame, time.time()])
        
        print("That makes %i new frames!" % newFrames)

        self.frameHistory.sort(key=lambda x: x[1])

        maxFrameBufferSize = 0

        if iteration < 5:
            maxFrameBufferSize = self.framesPerIteration * 5
        else:
            maxFrameBufferSize = self.framesPerIteration * 5 + (iteration - 5) * self.framesPerIteration // 2
            
        if maxFrameBufferSize > self.learner.learner.getFramesBufferSize():
            maxFrameBufferSize = self.learner.learner.getFramesBufferSize()

        print("Max frame buffer size for iteration %i is %i, currently used: %i" % (iteration, maxFrameBufferSize, len(self.frameHistory)))

        rmFrames = 0

        while len(self.frameHistory) > maxFrameBufferSize:
            rmFrames += 1
            del self.frameHistory[0]

        if rmFrames > 0:
            print("Removed %i old frames" % rmFrames)

        random.shuffle(self.frameHistory)
        
    
    def doLearningIteration(self, iteration, keepFramesPerc=1.0):
        t = time.time()
        t0 = t
        
        maxFrames = self.framesPerIteration
        
        framesPerProc = int(maxFrames / self.threads)
        
        print("Frames per process: " + str(framesPerProc))
        sys.stdout.flush()
        
        asyncs = []
        
        # for cProfile switch off
        useMp = True
        
        for tidx in range(self.threads):
            if self.useTreeFrameGeneration:
                if useMp:
                    asyncs.append(self.pool.apply_async(self.learner.selfPlayGamesAsTree, args=(framesPerProc, self.byThreadTreeGenerators[tidx])))
                else:
                    asyncs.append(self.learner.selfPlayGamesAsTree(framesPerProc, self.byThreadTreeGenerators[tidx])[0])
            else:
                if useMp:
                    asyncs.append(self.pool.apply_async(self.learner.selfPlayNFrames, args=(framesPerProc, self.batchSize, keepFramesPerc)))
                else:
                    asyncs.append(self.learner.selfPlayNFrames(framesPerProc, self.batchSize, keepFramesPerc))
         
        self.byThreadTreeGenerators = []
#         
        cframes = 0
        ignoreFrames = 0
        newFrames = []
         
        for asy in asyncs:
            asyResult = None
            if useMp:
                asyResult = asy.get()
                 
                if self.useTreeFrameGeneration:
                    self.byThreadTreeGenerators.append(asyResult[1])
                    asyResult = asyResult[0]
            else:
                asyResult = asy
            for f in asyResult:
                cframes += 1
                newFrames.append(f)
        
        print("Collected %i frames in %f" % (cframes, (time.time() - t)))
        
        # TODO make an abstracted method to ignore frames for reasons unknown to this code
        if ignoreFrames > 0:
            print("Ignored %i frames" % ignoreFrames)
        
        sys.stdout.flush()
        
        self.addNewFramesToHistory(newFrames, iteration)
        
        self.learnFrames([x[0] for x in self.frameHistory], iteration)

        didBenchmark = False

        if time.time() - self.lastBenchmarkTime > self.benchmarkTime:
            print("Benchmarking progress...")
            sys.stdout.flush()
            if self.benchmarkShowsProgress():
                self.bestIteration = iteration
            self.lastBenchmarkTime = time.time()
            didBenchmark = True

        print("Iteration completed in %f" % (time.time() - t0))
        
        sys.stdout.flush()
        
        return didBenchmark

    def learnFrames(self, learnFrames, iteration):
        t = time.time()
        
        random.shuffle(learnFrames)
        
        self.learner.learner.learnFromFrames(learnFrames, iteration, reAugmentEvery=self.reAugmentEvery, threads=self.threads)
        
        print("Done learning in %f" % (time.time() - t))

    def load(self, loadFrames = True, iteration = None):
        spath = os.path.join(self.workingdir, "status.json")
        if (os.path.exists(spath)):
            status = openJson(spath)
            
            if iteration is None:
                iteration = status["iteration"]
            
            print("Continue training of a player at iteration " + str(iteration))
        else:
            print("Beginning training of a new player")
            status = {}
            status["lastBenchmark"] = 0
            status["iteration"] = 0
            status["bestIteration"] = 0
            iteration = 0
        
        self.bestIteration = status["bestIteration"]
        self.lastBenchmarkTime = time.time() - status["lastBenchmark"]
        
        lPath = os.path.join(self.workingdir, "learner.iter" + str(iteration))
        if os.path.exists(lPath):
            self.learner.learner.initState(lPath)
        
        fPath = os.path.join(self.workingdir, "frameHistory"+ str(iteration) +".pickle")
        if os.path.exists(fPath) and loadFrames:
            with open(fPath, "rb") as f:
                self.frameHistory = pickle.load(f)
                print("Loaded %i frames " % len(self.frameHistory))
            
        return iteration + 1
    
    def saveFrames(self, iteration):
        fPath = os.path.join(self.workingdir, "frameHistory"+ str(iteration) +".pickle")
        with open(fPath, "wb") as f:
            pickle.dump(self.frameHistory, f)
            print("Saved %i frames for iteration %i" % (len(self.frameHistory), iteration))
    
    def saveForIteration(self, iteration):
        self.saveFrames(iteration)
        self.learner.learner.saveState(os.path.join(self.workingdir, "learner.iter" + str(iteration)))
        
        status = {}
        status["lastBenchmark"] = time.time() - self.lastBenchmarkTime
        status["iteration"] = iteration
        status["bestIteration"] = self.bestIteration
        
        writeJson(os.path.join(self.workingdir, "status.json"), status)
    
    def iterateLearning(self, iterateForever = True, keepFramesPerc=1.0):
        i = self.load()
        
        while True:
            print("Begin iteration %i @ %s" % (i, time.ctime()))
            didBenchmark = self.doLearningIteration(i, keepFramesPerc=keepFramesPerc)

            self.saveForIteration(i)
            i += 1
            
            if didBenchmark and (not iterateForever):
                break
            