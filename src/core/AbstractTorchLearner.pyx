# cython: profile=True

'''
Created on Oct 27, 2017

@author: cclausen
'''


import abc

import torch
from torch.autograd import Variable
import numpy as np

from core.AbstractLearner import AbstractLearner

import random

import time

import multiprocessing as mp

def fillTrainingSetPart0(gameInit, playerCount, frames, startIndex, moveOutput, winOutput, networkInput):
    moveOutput[startIndex : startIndex + len(frames)].fill_(0)
    winOutput[startIndex : startIndex + len(frames)].fill_(0)
    networkInput[startIndex : startIndex + len(frames)].fill_(0)
    
    for fidx, frame in enumerate(frames):
        augmented = frame[0].augmentFrame(frame)
        
        gameInit.fillNetworkInput(augmented[0], networkInput, startIndex + fidx)
        
        for idx, p in enumerate(augmented[1]):
            moveOutput[startIndex + fidx, idx] = p
        
        for pid in range(playerCount):
            winOutput[startIndex + fidx, augmented[0].mapPlayerIndexToTurnRel(pid)] = frame[3][pid]

class AbstractTorchLearner(AbstractLearner, metaclass=abc.ABCMeta):
    def __init__(self, framesBufferSize, batchSize, epochs, lr_schedule):
        assert framesBufferSize % batchSize == 0

        self.lr_schedule = lr_schedule
        self.framesBufferSize = framesBufferSize
        self.batchSize = batchSize
        self.epochs = epochs
        self.netInIsCached = False
        self.netInCache = None
        self.netInCacheCpu = None
    
    def getFramesBufferSize(self):
        return self.framesBufferSize
    
    @abc.abstractmethod
    def getNetInputShape(self):
        """
        returns a tuple that describes the input shape of the network, minus the batchdimension
        """
    
    @abc.abstractmethod
    def getPlayerCount(self):
        """
        returns the number of players
        """

    @abc.abstractmethod
    def getMoveCount(self):
        """
        returns the number of possible moves a player can make
        """
    
    def getLrForIteration(self, iteration):
        """
        return the learning rate to be used for the given iteration
        """
        if iteration >= len(self.lr_schedule):
            iteration = len(self.lr_schedule) - 1
        return self.lr_schedule[iteration]
    
    def getBatchSize(self):
        return self.batchSize
    
    @abc.abstractmethod
    def createNetwork(self):
        """
        return a newly created Torch Network
        """
    
    @abc.abstractmethod
    def createOptimizer(self, net):
        """
        return a torch optimizer to be used for the learning process
        """
    
    @abc.abstractmethod
    def fillNetworkInput(self, state, tensor, batchIndex):
        """
        fill the given tensor with the input that represents state at the given batchIndex.
        The tensor is zero'd out before this is called
        """
    
    def initState(self, file):
        self.networkInput = torch.zeros((self.framesBufferSize,) + self.getNetInputShape())
        self.moveOutput = torch.zeros(self.framesBufferSize, self.getMoveCount())
        self.winOutput = torch.zeros(self.framesBufferSize, self.getPlayerCount())

        self.net = self.createNetwork()
        self.opt = self.createOptimizer(self.net)
        
        if file != None:
            self.net.load_state_dict(torch.load(file))
            print("Loaded state from " + file)
        
        self.net.cuda()
        self.net.train(False)
    
    def saveState(self, file):
        torch.save(self.net.state_dict(), file)
    
    def initGpuStateCache(self):
        self.gpuCacheSize = 64
        self.gpuCacheElemCount = 0
        self.gpuCacheTimeIndex = 0
        self.gpuCacheAges = [0] * self.gpuCacheSize
        self.cpuStagingArea = torch.zeros((1, ) + self.getNetInputShape()).pin_memory()
        self.gpuCache = torch.zeros((self.gpuCacheSize, ) + self.getNetInputShape()).cuda()
        self.gpuCacheMapping = {}
        
    # this does not appear to help performance :(
    def fillQuick(self, int idx, object state):
        if state.id in self.gpuCacheMapping:
#             print("Perfect match")
            self.netInCache[idx] = self.gpuCache[self.gpuCacheMapping[state.id]]
        elif state.lastId in self.gpuCacheMapping:
#             print("Partial match")
            gidx = self.gpuCacheMapping[state.lastId]
            self.netInCache[idx] = self.gpuCache[gidx]
            self.gpuCacheAges[gidx] = self.gpuCacheTimeIndex
            state.invertTensorField(self.netInCache, idx)
            state.updateTensorForLastMove(self.netInCache, idx)
        else:
#             print("Cache miss")
            self.fillNetworkInput(state, self.cpuStagingArea, 0)
            self.netInCache[idx] = self.cpuStagingArea[0]
            
            oldestAge = self.gpuCacheTimeIndex
            oldestIndex = 0
            for gidx in range(self.gpuCacheSize):
                gAge = self.gpuCacheAges[gidx] 
                if gAge  < oldestAge:
                    oldestAge = gAge
                    oldestIndex = gidx
            
            self.gpuCacheMapping[state.id] = oldestIndex
            self.gpuCacheAges[oldestIndex] = self.gpuCacheTimeIndex
            self.gpuCache[oldestIndex] = self.netInCache[idx]
    
    """
    this has to be able to deal with None values in the batch!
    also if len(batch) > batchSize this will explode
    """
    def evaluate(self, batch):
        if not self.netInIsCached:#
            self.netInCacheCpu = torch.zeros((self.batchSize, ) + self.getNetInputShape()).pin_memory()
            self.netInCache = Variable(torch.zeros((self.batchSize, ) + self.getNetInputShape())).cuda()
            self.netInIsCached = True
#             self.initGpuStateCache()

#         self.gpuCacheTimeIndex += 1

        cdef int batchSize = len(batch)
        cdef int idx, bidx
        
        for idx in range(batchSize):
            b = batch[idx]
            if b is not None:
                state = b
#                 self.fillQuick(idx, state)
                
                self.fillNetworkInput(state, self.netInCacheCpu, idx)

        self.netInCache[:len(batch)] = self.netInCacheCpu[:len(batch)]

#         assert np.all(self.netInCache[:len(batch)].cpu().numpy() == self.netInCacheCpu[:len(batch)])
#         print("OK", batchSize)
        
        moveP, winP = self.net(self.netInCache[:len(batch)])

        winP = torch.exp(winP)
        moveP = torch.exp(moveP)
        
        cdef int pcount = state.getPlayerCount()
        cdef int pid
        
        results = []
        for bidx in range(batchSize):
            b = batch[bidx]
            if b is not None:
                state = b
                
                r = moveP.data[bidx]
                assert r.is_cuda #this is here because a copy is needed and I want to make sure r is gpu, so cpu() yields a copy
                r = r.cpu()
                
                w = []
                for pid in range(pcount):
                    w.append(winP.data[bidx, state.mapPlayerIndexToTurnRel(pid)].item())

                results.append((r, w))
            else:
                results.append(None) #assumed to never be read. None is a pretty good bet to make everything explode

        return results
    
    def fillTrainingSet(self, frames):
        random.shuffle(frames)
        
        self.moveOutput.fill_(0)
        self.winOutput.fill_(0)
        self.networkInput.fill_(0)
        
        for fidx, frame in enumerate(frames):
            
            augmented = frame[0].augmentFrame(frame)
            
            self.fillNetworkInput(augmented[0], self.networkInput, fidx)
            
            for idx, p in enumerate(augmented[1]):
                self.moveOutput[fidx, idx] = p
            
            for pid in range(self.getPlayerCount()):
                self.winOutput[fidx, augmented[0].mapPlayerIndexToTurnRel(pid)] = frame[3][pid]
    

#     def fillTrainingSetPart(self, frames, startIndex, moveOutput, winOutput, networkInput):
#         moveOutput[startIndex : startIndex + len(frames)].fill_(0)
#         winOutput[startIndex : startIndex + len(frames)].fill_(0)
#         networkInput[startIndex : startIndex + len(frames)].fill_(0)
#         
#         for fidx, frame in enumerate(frames):
#             augmented = frame[0].augmentFrame(frame)
#             
#             self.fillNetworkInput(augmented[0], networkInput, startIndex + fidx)
#             
#             for idx, p in enumerate(augmented[1]):
#                 moveOutput[startIndex + fidx, idx] = p
#             
#             for pid in range(self.getPlayerCount()):
#                 winOutput[startIndex + fidx, augmented[0].mapPlayerIndexToTurnRel(pid)] = frame[3][pid]
    
    def learnFromFrames(self, frames, iteration, dbg=False, reAugmentEvery=1, threads=4):
        assert(len(frames) <= self.framesBufferSize), str(len(frames)) + "/" + str(self.framesBufferSize)

        batchNum = int(len(frames) / self.batchSize)
        
        if dbg:
            print(len(frames), self.batchSize, batchNum)
                
        lr = self.getLrForIteration(iteration)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
        
        print("learning rate for iteration %i is %f" % (iteration, lr))
        
        # the model is in non-training mode by default, as set by initState
        self.net.train(True)
        
        cdef int e, bi
        
        pool = mp.Pool(processes = threads)
        
        for e in range(self.epochs):
            print("Preparing example data...")
            if e % reAugmentEvery == 0:
                print("Filling with augmented data")
                
                # self.fillTrainingSet(frames)
                
                asyncs = []
                framesPerProc = int(len(frames) / threads)
                for i in range(threads):
                    startIndex = framesPerProc * i
                    endIndex = startIndex + framesPerProc
                    if i == threads - 1:
                        endIndex = len(frames)
                    
                    asyncs.append(pool.apply_async(fillTrainingSetPart0, args=(self.getGameInit(), self.getPlayerCount(), frames[startIndex:endIndex], startIndex, self.moveOutput, self.winOutput, self.networkInput)))
                
                for asy in asyncs:
                    asy.get()
                
#                 print(self.networkInput[0])
#                 print(self.moveOutput)
#                 print(self.winOutput)
                
                assert torch.sum(self.networkInput.ne(self.networkInput)) == 0
                assert torch.sum(self.moveOutput.ne(self.moveOutput)) == 0
                assert torch.sum(self.winOutput.ne(self.winOutput)) == 0
                 
                lf = len(frames)
                nIn = Variable(self.networkInput[:lf]).cuda()
                mT = Variable(self.moveOutput[:lf]).cuda()
                wT = Variable(self.winOutput[:lf]).cuda()
            else:
                print("Shuffle last augmented data...")
                perm = torch.randperm(len(frames)).cuda()
                nInR = nIn.index_select(0, perm)
                mTR = mT.index_select(0, perm)
                wTR = wT.index_select(0, perm)
                del nIn
                del mT
                del wT
                nIn = nInR
                mT = mTR
                wT = wTR
            print("Data prepared, starting to learn!")
            
            mls = []
            wls = []
            
            for bi in range(batchNum):
                x = nIn[bi*self.batchSize : (bi+1) * self.batchSize]
                yM = mT[bi*self.batchSize : (bi+1) * self.batchSize]
                yW = wT[bi*self.batchSize : (bi+1) * self.batchSize]

                self.opt.zero_grad()
                
                if dbg:
                    print(x, yM, yW)
                
                mO, wO = self.net(x)
                
                mLoss = -torch.sum(mO * yM) / self.batchSize
                wLoss = -torch.sum(wO * yW) / self.batchSize
                
                loss = mLoss + wLoss
                loss.backward()
                
                # TODO maybe use some gradient clipping to be save?
                
                self.opt.step()
                
                mls.append(mLoss.data.item())
                wls.append(wLoss.data.item())
                
            print("Completed Epoch %i with loss %f + %f" % (e, np.mean(mls), np.mean(wls)))
        
        self.net.train(False)
        
        del nIn
        del mT
        del wT
        
        pool.terminate()

