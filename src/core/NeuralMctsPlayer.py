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

# very very slow
#from core.MctsTree import TreeNode, batchedMcts

# 10x speed to the above, "score of 1.87"
#from core.cMctsTree import TreeNode, batchedMcts  # @UnresolvedImport

# not bad for a pure python thing
#from core.vectorMcts import TreeNode, batchedMcts 

# the best: "score of 4.47". Not yet implemented: move grouping, not sure how relevant that would still be
from core.cvectorMcts import TreeNode, batchedMcts # @UnresolvedImport

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

def padRight(sx, n):
    while len(sx) < n:
        sx = sx + " "
    return sx
    
def alignStringBlocks(blockA, blockB):
    blockA = blockA.splitlines()
    blockB = blockB.splitlines()
    
    maxLen = max(max([len(x) for x in blockA]), max([len(x) for x in blockB]))
    
    lineNum = max([len(blockA), len(blockB)])
    
    result = ""
    
    for i in range(lineNum):
        a = ""
        b = ""
        if len(blockA) > i:
            a = blockA[i]
        if len(blockB) > i:
            b = blockB[i]
        result += str(padRight(a, maxLen))
        result += "   "
        result += str(padRight(b, maxLen))
        result += "\n"
    
    return result
    

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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class NeuralMctsPlayer():
    def __init__(self, stateTemplate, config, learner):
        self.config = config
        self.stateTemplate = stateTemplate.clone()
        self.mctsExpansions = config["learning"]["mctsExpansions"] # a value of 1 here will make it basically play by the network probabilities in a greedy way #TODO test that
        self.learner = learner
        self.cpuct = 2.75
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
        hasBetterThanXMoves = False
        x = 0.03
        eps = 0.0000001
        for idx, p in enumerate(moveP):
            if state.isMoveLegal(idx) and p > 0:
                possibleMoves += 1
                
                psum += p + eps
                
                ms.append(idx)
                ps.append(p + eps)

        assert len(ms) > 0, "The state should have legal moves that were considered"
        
        if possibleMoves <= count:
            return ms

        self.shuffleTogether(ms, ps)

        for idx in range(len(ps)):
            ps[idx] /= float(psum)
            
            if ps[idx] > x:
                hasBetterThanXMoves = True
        
        if explore:
            if hasBetterThanXMoves:
                psum = 0
                for idx in range(len(ps)):
                    if ps[idx] < x:
                        if count > 1:
                            ps[idx] = eps
                        else:
                            ps[idx] = 0
                    psum += ps[idx]
                    
                for idx in range(len(ps)):
                    ps[idx] /= float(psum)
                    
            m = np.random.choice(ms, count, replace=False, p = ps)
        else:
            if count > 1:
                m = []
                for arg in getBestNArgs(count, ps):
                    m.append(ms[arg])
            else:
                m = [ms[np.argmax(ps)]]
        return m

    def _pickMovesY(self, count, moveP, state, explore=False):
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
    
    def evaluateByLearner(self, states, asyncCall=None):
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
            eResults = self.learner.evaluate(packed, asyncCall)
        else:
            if asyncCall is not None:
                asyncCall()
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
        
        def evaCall(ws, asyncCall = None):
            return self.evaluateByLearner(ws, asyncCall)
        
        batchedMcts(states, self.mctsExpansions, evaCall, self.cpuct)
        
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
    def findBestMoves(self, states, noiseMix=0.25):
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
        
        gamesPlayed = 0
        draws = 0
        
        while len(frames) < n:
            #print("Begin batch")
            #t = time.time()
            self.batchMcts(batch, useAdvantagePlayer = False)
            #print("Batch complete in %f" % (time.time() - t))
            
            for idx in range(batchSize):
                b = batch[idx]
                assert b != None
                md = b.getMoveDistribution()
                if b.state.getTurn() > 0:
                    bframes[idx].append([b.state.getFrameClone(), md, b.getBestValue(), b.getEdgePriors(), b.getNetValueEvaluation()])

                mv = self._pickMoves(1, md, b.state, b.state.isEarlyGame())[0]
                b = b.getChildForMove(mv)
                
                if b.state.isTerminal():
                    printGame = gamesPlayed % 50 == 0
                    
                    gamesPlayed += 1
                    termResult = b.getTerminalResult()
                    
                    if b.state.getWinner() == -1:
                        draws += 1
                    
                    if printGame:
                        dStr = "Completed game " + str(gamesPlayed) + ", it went as follows with win target " + str(termResult) + "\n" 
                    
                    for f in bframes[idx]:
                        frames.append(f[:-2] + [b.getTerminalResult()])
                        ancestor = frames[-1]
                        
                        if printGame:
                            treeSearchChoices = "Improved moves\n" + ancestor[0].moveProbsAndDisplay(ancestor[1])
                            netChoices = "Network moves, Network values are " + str(f[-1]) + "\n" + ancestor[0].moveProbsAndDisplay(f[-2])
                            sframe = alignStringBlocks(treeSearchChoices, netChoices) + "\n"
                            dStr += sframe
                    
                    if printGame:
                        print(dStr)
                    
                    bframes[idx] = []
                    batch[idx] = TreeNode(self.stateTemplate.getNewGame())
                else:
                    batch[idx] = b
                
        print("Completed %i games with %i draws" % (gamesPlayed, draws))
                
        torch.cuda.empty_cache()
        return frames
        
    # play one game like in training, print out the game with probs 'n' stuff, return a package of all states encountered that contains detailed info on the 
    # search trees used for each step to debug them
    def recordDemoGame(self):
        trees = []
        
        gameNode = TreeNode(self.stateTemplate.getNewGame())

        while not gameNode.state.isTerminal():
            self.batchMcts([gameNode])
            trees.append(gameNode.exportTree())
            
            md = gameNode.getMoveDistribution()
            mv = self._pickMoves(1, md, gameNode.state, gameNode.state.isEarlyGame())[0]
            
            leftPart = ("Improved moves using max depth %i\n" % gameNode.getTreeDepth()) + gameNode.state.moveProbsAndDisplay(md)
            rightPart = "Network moves, Network values are " + str(gameNode.getNetValueEvaluation()) + "\n" + gameNode.state.moveProbsAndDisplay(gameNode.getEdgePriors())
            
            print(alignStringBlocks(leftPart, rightPart))
            
            gameNode = gameNode.getChildForMove(mv)
        
        return trees
        
    def getPlayFunc(self):
        gameState = self.stateTemplate.getNewGame()
        
        def playFunc(movesList):
            nonlocal gameState
            
            for move in movesList:
                gameState.simulate(move)
                print(str(gameState))
                if gameState.isTerminal():
                    return (gameState.getWinner(), None)
            myId = gameState.getPlayerOnTurnIndex()
            myMoves = []
            while gameState.getPlayerOnTurnIndex() == myId:
                move = self.findBestMoves([gameState])[0]
                gameState.simulate(move)
                print(str(gameState))
                if gameState.isTerminal():
                    return (gameState.getWinner(), None)
                myMoves.append(move)
            return (None, myMoves)
        
        return playFunc
        
        
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
        
    