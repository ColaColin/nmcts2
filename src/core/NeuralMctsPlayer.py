'''

A class that takes an AbstractLearner and offers functions to play:
- against other NeuralMctsPlayers in large batches
- against humans
- generally a function to find the best move in a given situation, in batches
- used to generate lots of game frames to learn from


This class is where the actual MCTS guided by the neural network happens,
in batches of games which translate to batches of situations to be evaluated at once for the NN

Created on Oct 27, 2017

@author: cclausen
'''

import numpy as np

#from core.MctsTree import TreeNode, batchedMcts
# 10x speed
from core.cMctsTree import TreeNode, batchedMcts  # @UnresolvedImport

import random

import time

import math

import os

import sys

import torch.cuda

def hconc(strs):
    result = ""
    
    if len(strs) == 0:
        return result
    
    strs = [s.split("\n") for s in strs]
    maxHeight = max([len(s) for s in strs])
    
    for i in range(maxHeight):
        for s in strs:
            result += s[i]
            result += " | "
        result += "\n"
    
    return result

class TreeFrameGenerator():
    
    def __init__(self, batchSize, poolSize):
        self.runningGames = []
        self.unfinalizedFrames = []
        self.roots = 0
        
        self.targetGamesCount = poolSize
        self.batchSize = batchSize
        
    def mcts(self, player):
        # do mcts searches for all current states            
        batchCount = math.ceil(len(self.runningGames) / self.batchSize)
        for idx in range(batchCount):
            startIdx = idx * self.batchSize
            endIdx = (idx+1) * self.batchSize
            batch = self.runningGames[startIdx:min(endIdx, len(self.runningGames)+1)]
            player.batchMcts(batch, useAdvantagePlayer = True)
    
    def printRunningGames(self):
        strs = [str(g.state) for g in self.runningGames]
        print(hconc(strs))
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
    
    def generateNextN(self, n, player):
        finalizedFrames = []
        lastLog = 5
        forceLog = True

        advantagePlayerWins = 0
        disadvantagePlayerWins = 0
        drawCount = 0

        while len(finalizedFrames) < n:
            
            #self.printRunningGames()
            
            if forceLog or lastLog <= len(finalizedFrames):
                print("[Process#%i] Collected %i / %i, running %i games with %i roots and %i draws, advantagePlayerWins %i, disadvantagePlayerWins %i" % (os.getpid(), 
                        len(finalizedFrames), n, len(self.runningGames), self.roots, drawCount, advantagePlayerWins, disadvantagePlayerWins))
                if not forceLog:
                    lastLog = 100 + len(finalizedFrames)
                sys.stdout.flush()
                forceLog = False
            
            currentGameCount = len(self.runningGames)
            targetGamesCount = self.targetGamesCount

            if self.roots < targetGamesCount // 5 and (currentGameCount < 3 or (targetGamesCount > currentGameCount and currentGameCount % 5 == 0)):
                self.runningGames.append(TreeNode(player.stateTemplate.getNewGame()))
                self.unfinalizedFrames.append(None)
                currentGameCount += 1
            
            newRunningGames = []
            newUnfinalFrames = []
            
            self.mcts(player)
            
            if (targetGamesCount > currentGameCount):
                splits = (targetGamesCount - currentGameCount)//5
                
                splitIdx = random.randint(0, currentGameCount)
                splitIdxs = []
                for i in range(splits):
                    offset = 0
                    for offset in range(currentGameCount + 1):
                        nextIdx = (splitIdx + i + offset) % currentGameCount
                        if self.unfinalizedFrames[nextIdx] != None and (offset >= currentGameCount or (self.unfinalizedFrames[nextIdx][6] > 2 and self.unfinalizedFrames[nextIdx][5] < 4)):
                            splitIdxs.append(nextIdx)
                            break

            for gidx in range(currentGameCount):
                g = self.runningGames[gidx]
                md = g.getMoveDistribution()

                splitExtra = 0
                for sIdx in splitIdxs:
                    if gidx == sIdx:
                        splitExtra += 1
                        break

                mvs = player._pickMoves(1 + splitExtra, md, g.state, g.state.isEarlyGame())
                
                parentStateFrame = self.unfinalizedFrames[gidx]

                ancestor = parentStateFrame
                while ancestor is not None:
                    ancestor[5] += len(mvs) - 1
                    ancestor = ancestor[4]
                
                if g.state.getTurn() > 0:
                    isSplitting = len(mvs) > 1
                    turnsSinceSplit = 0
                    
                    if not isSplitting and parentStateFrame is not None:
                        turnsSinceSplit = parentStateFrame[6] + 1
                    
                    stateFrame = [g.state.getFrameClone(),
                                  md,
                                  g.getBestValue(),
                                  np.array([0.0] * g.state.getPlayerCount()),
                                  parentStateFrame,
                                  len(mvs),
                                  turnsSinceSplit]
                else:
                    stateFrame = None
                
                for mv in mvs:
                    nextState = g.getChildForMove(mv)
                    
                    if nextState.state.getTurn() == 1:
                        self.roots += 1
                    
                    if not nextState.state.isTerminal():
                        newRunningGames.append(nextState)
                        newUnfinalFrames.append(stateFrame)
                    else:
                        stateResult = np.array(nextState.getTerminalResult())
                        
                        ancestor = stateFrame
                        depth = 0
                        while ancestor is not None:
                            ancestor[3] += stateResult
                            depth += 1
                            if np.sum(ancestor[3]) > ancestor[5] - 0.5:
                                eps = 0.000000001
                                ancestor[3] = (ancestor[3]) / (np.sum(ancestor[3]) + eps)
                                if depth <= nextState.state.getPlayerCount() or ancestor[5] > 1:
                                    finalizedFrames.append(ancestor[:4])
                                    if nextState.state.getWinner() == -1:
                                        drawCount += 1

                                if ancestor[4] is None:
                                    self.roots -= 1
                                    assert self.roots > -1
                            ancestor = ancestor[4]
                            
            
            self.runningGames = newRunningGames
            self.unfinalizedFrames = newUnfinalFrames
        
        print("Completed collecting frames with a draw rate ", drawCount / len(finalizedFrames))
        
        return finalizedFrames


def getBestNArgs(n, array):
    if len(array) < n:
        return np.arange(0, len(array))
    
    result = [np.argmin(array)] * n
    for idx, pv in enumerate(array):
        rIdx = 0
        isAmongBest = False
        while rIdx < n and pv > array[result[rIdx]]:
            isAmongBest = True
            rIdx += 1
            
        if isAmongBest:
            if rIdx > 1:
                for fp in range(rIdx-1):
                    result[fp] = result[fp+1]
            result[rIdx-1] = idx
    
    return result

class NeuralMctsPlayer():
    def __init__(self, stateTemplate, config, learner):
        self.config = config
        self.stateTemplate = stateTemplate.clone()
        self.mctsExpansions = config["learning"]["mctsExpansions"] # a value of 1 here will make it basically play by the network probabilities in a greedy way #TODO test that
        self.learner = learner
        self.cpuct = 0.5424242 #hmm TODO: investigate the influence of this factor on the speed of learning
        self.batchMctsCalls = 0
        self.batchMctsTime = 0
        self.lastBatchMctsBenchmarkTime = time.time()
        self.unbalanceTrainingMctsFactor = config["learning"]["unbalanceTrainingMctsFactor"]

    def clone(self):
        return NeuralMctsPlayer(self.stateTemplate, self.config, 
                                self.learner.clone())

    def shuffleTogether(self, a, b):
        combined = list(zip(a, b))
        random.shuffle(combined)
        a[:], b[:] = zip(*combined)

    def _pickMoves(self, count, moveP, state, explore=False):
        ms = []
        ps = []
        psum = 0.0
        possibleMoves = 0
        for idx, p in enumerate(moveP):
            if state.isMoveLegal(idx):
                possibleMoves += 1
                eps = 0.0001
                psum += p + eps
                
                ms.append(idx)
                ps.append(p + eps)

        assert len(ms) > 0, "The state should have legal moves"
        
        if possibleMoves <= count:
            return ms

        self.shuffleTogether(ms, ps)

        for idx in range(len(ps)):
            ps[idx] /= float(psum)
        
        if explore:
            m = np.random.choice(ms, count, replace=False, p = ps)
        else:
            if count > 1:
                m = []
                for arg in getBestNArgs(count, ps):
                    m.append(ms[arg])
            else:
                m = [ms[np.argmax(ps)]]
        return m

    def isRelevantEvaluationInput(self, s):
        return s != None and not s.isTerminal()

    def getNonRelevantEvaluationResult(self, s):
        return (None, s.getTerminalResult())
    
    def evaluateByLearner(self, states):
        packed = []
        packIdxMap = {}
        hasInput = False
        
        for idx, s in enumerate(states):
            if s != None and not s.state.isTerminal():
                hasInput = True
                packed.append(s.state)
                pidx = len(packed)-1
                packIdxMap[idx] = pidx
        
        if hasInput:
            eResults = self.learner.evaluate(packed)
        else:
            eResults = []
        
        finalResults = [None] * len(states)

        for idx in range(len(states)):
            if states[idx] != None:
                if (states[idx].state.isTerminal()):
                    finalResults[idx] = (None, states[idx].getTerminalResult())
                else:
                    finalResults[idx] = eResults[packIdxMap[idx]]
        
        return finalResults

    def getBatchMctsResults(self, frames, startIndex):
        nodes = [TreeNode(n[0]) for n in frames]
        self.batchMcts(nodes, useAdvantagePlayer = False)
        result = []
        for idx, f in enumerate(nodes):
            result.append(( startIndex + idx, f.getMoveDistribution(), f.getBestValue() ))
        return result
    
    def batchMcts(self, states, useAdvantagePlayer = False):
        """
        runs batched mcts guided by the learner
        yields a result for each state in the batch
        states is expected to be an array of TreeNode(state)
        those TreeNodes will be changed as a result of the call
        """
        assert self.mctsExpansions > 0
        
        t = time.time()
        
        batchedMcts(states, self.mctsExpansions, lambda ws: self.evaluateByLearner(ws), self.cpuct)
        
        self.batchMctsTime += (time.time() - t)
        self.batchMctsCalls += len(states)
        
        if self.lastBatchMctsBenchmarkTime < (time.time() - 60 * 3):
            self.lastBatchMctsBenchmarkTime = time.time()
            bt= self.batchMctsCalls / self.batchMctsTime
            self.batchMctsTime = 0
            self.batchMctsCalls = 0
            print("[Process#%i]: %f moves per second " % (os.getpid(), bt))
            sys.stdout.flush()
    
    # todo if one could get the caller to deal with the treenode data it might be possible to not throw away the whole tree that was build, increasing play strength
    def findBestMoves(self, states, noiseMix=0.2):
        """
        searches for the best moves to play in the given states
        this means the move with the most visits in the mcts result greedily is selected
        states is expected to be an array of game states. TreeNodes will be put around them.
        the result is an array of moveIndices
        """
        ts = [TreeNode(s, noiseMix=noiseMix) if s != None else None for s in states]
        self.batchMcts(ts, useAdvantagePlayer = False)
        bmoves = [self._pickMoves(1, s.getMoveDistribution(), s.state, False)[0] if s != None else None for s in ts]
        return bmoves
        
    def playVsHuman(self, state, humanIndex, otherPlayers, stateFormatter, commandParser):
        """
        play a game from the given state vs a human using the given command parser function,
        which given a string from input is expected to return a valid move or -1 as a means to signal
        an invalid input.
        stateFormatter is expected to be a function that returns a string given a state
        """
        allPlayers = [self] + otherPlayers
        if humanIndex != -1:
            allPlayers.insert(humanIndex, None)
        
        print("You are player " + str(humanIndex))
        
        while not state.isTerminal():
            print(stateFormatter(state))
            pindex = state.getPlayerOnTurnIndex()
            player = allPlayers[pindex]
            if player != None: #AI player
                m = player.findBestMoves([state])[0]
            else:
                m = -1
                while m == -1:
                    m = commandParser(input("Your turn:"))
                    if m == -1:
                        print("That cannot be parsed, try again.")
                    if not state.isMoveLegal(m):
                        print("That move is illegal, try again.")
                        m = -1
            state.simulate(m)
            
        print("Game over")
        print(stateFormatter(state))
        
    def selfPlayGamesAsTree(self, collectFrameCount, treeGen = None):
        torch.cuda.empty_cache() #this is mainly here to verify it won't crash before spending a few hours generating frames...
        if treeGen is None:
            treeGen = TreeFrameGenerator(self.config["learning"]["batchSize"], self.config["learning"]["treePoolSize"])
        
        self.lastBatchMctsBenchmarkTime = time.time()
        frames = treeGen.generateNextN(collectFrameCount, self) 
        torch.cuda.empty_cache()
        return frames, None #fck python...
        
    def selfPlayNFrames(self, n, batchSize, keepFramesPerc):
        self.lastBatchMctsBenchmarkTime = time.time()
        """
        plays games until n frames are collected against itself using more extensive exploration (i.e. pick move probabilistic if state reports early game)
        used to generate games for playing.
        """
        frames = []
        
        batch = []
        bframes = []
        for _ in range(batchSize):
            initialGameState = self.stateTemplate.getNewGame()
            batch.append(TreeNode(initialGameState))
            bframes.append([])
        
        while len(frames) < n:
            print("Begin batch")
            t = time.time()
            self.batchMcts(batch, useAdvantagePlayer = True)
            print("Batch complete in %f" % (time.time() - t))
            
            for idx in range(batchSize):
                b = batch[idx]
                if b == None:
                    continue
                md = b.getMoveDistribution()
                if b.state.getTurn() > 0:
                    bframes[idx].append([b.state.getFrameClone(), md, b.getBestValue()])
                mv = self._pickMoves(1, md, b.state, b.state.isEarlyGame())[0]
                b = b.getChildForMove(mv)
                
                if b.state.isTerminal():
                    for f in bframes[idx]:
                        if keepFramesPerc == 1.0 or random.random() < keepFramesPerc:
                            frames.append(f + [b.getTerminalResult()])
                    bframes[idx] = []
                    batch[idx] = TreeNode(self.stateTemplate.getNewGame(), useAdvantagePlayer = True)
                else:
                    batch[idx] = b
                
                
        torch.cuda.empty_cache()
        return frames
        
    def playAgainst(self, n, batchSize, others, collectFrames=False):
        """
        play against other neural mcts players, in batches.
        Since multiple players are used this requires more of a lock-step kind of approach, which makes
        it less efficient than self play!
        returns a pair of:
            the number of wins and draws ordered as [self] + others with the last position representing draws
            a list of lists with the frames of all games, if collectFrames = True
        The overall number of players should fit with the game used.
        No exploration is done here.
        
        !!!Remember that if the game has a first move advantage than one call of this method is probably not enough to fairly compare two players!!!
        """
        
        assert n % batchSize == 0

        batchCount = int(n / batchSize)
        
        results = [0] * (2+len(others)) # the last index stands for draws, which are indexed with -1
        
        allPlayers = [self] + others
        
        gameFrames = []
        if collectFrames:
            for _ in range(n):
                gameFrames.append([])
        
        for bi in range(batchCount):
            
            gamesActive = 0
            batch = []
            
            for _ in range(batchSize):
                initialGameState = self.stateTemplate.getNewGame()
                batch.append(TreeNode(initialGameState))
                gamesActive += 1
            
            while gamesActive > 0:
                someGame = None
                for b in batch:
                    if b != None:
                        someGame = b
                        break
                # the games are all in the same turn
                pIndex = someGame.state.getPlayerOnTurnIndex()
                for b in batch:
                    if b != None:
                        assert b.state.getPlayerOnTurnIndex() == pIndex
                        
                player = allPlayers[pIndex]
                player.batchMcts(batch, useAdvantagePlayer = False)
                
#                 print(pIndex, someGame.state.c6, ["{0:.5f}".format(x) for x in someGame.getMoveDistribution()])
                
                newBatch = []
                
                for idx in range(len(batch)):
                    b = batch[idx]
                    md = b.getMoveDistribution()
                    
                    gameIndex = batchSize * bi + idx
                    
                    if collectFrames:
                        gameFrames[gameIndex].append([b.state.getFrameClone(), md, b.getBestValue()])
                    
                    mv = self._pickMoves(1, md, b.state, False)[0]
                    b = b.getChildForMove(mv)
                    
                    if b.state.isTerminal():
                        if collectFrames:
                            for f in gameFrames[gameIndex]:
                                f.append(b.getTerminalResult())
                                
                        gamesActive -= 1
                        results[b.state.getWinner()] += 1
                    else:
                        b.cutTree()
                        newBatch.append(b)
                    
                batch = newBatch

        sys.stdout.flush()

        return results, gameFrames
        
    