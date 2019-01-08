# cython: profile=False

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

import torch.cuda

def fillTrainingSetPart0(gameInit, int playerCount, frames, int startIndex, moveOutput, winOutput, networkInput):
    moveOutput[startIndex : startIndex + len(frames)].fill_(0)
    winOutput[startIndex : startIndex + len(frames)].fill_(0)
    networkInput[startIndex : startIndex + len(frames)].fill_(0)
    
    # not doing this slow down stuff by a factor of 6x... wow
    cdef float [:,:] moveTensor = moveOutput.numpy()
    cdef float [:,:] winTensor = winOutput.numpy()
    
    cdef int fidx, idx, pid
    cdef object frame
    cdef float p
    
    cdef int pTurnIdx
    
    for fidx, frame in enumerate(frames):
        augmented = frame[0].augmentFrame(frame)
        
        gameInit.fillNetworkInput(augmented[0], networkInput, startIndex + fidx)
        
        for idx, p in enumerate(augmented[1]):
            moveTensor[startIndex + fidx, idx] = p
        
        for pid in range(playerCount):
            pTurnIdx = augmented[0].mapPlayerIndexToTurnRel(pid)
            winTensor[startIndex + fidx, pTurnIdx] = frame[3][pid]

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
        # why even have two sets of buffers? the buffers are copied into vram anyway? hmmmmmm
        self.networkInputA = torch.zeros((self.framesBufferSize,) + self.getNetInputShape())
        self.networkInputB = torch.zeros((self.framesBufferSize,) + self.getNetInputShape())
        
        self.moveOutputA = torch.zeros(self.framesBufferSize, self.getMoveCount())
        self.moveOutputB = torch.zeros(self.framesBufferSize, self.getMoveCount())
        
        self.winOutputA = torch.zeros(self.framesBufferSize, self.getPlayerCount())
        self.winOutputB = torch.zeros(self.framesBufferSize, self.getPlayerCount())

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
    
    def queueFillBuffers(self, pool, frames, useBufferA, threads):
        random.shuffle(frames)
        
        if useBufferA:
            mo = self.moveOutputA
            wo = self.winOutputA
            ni = self.networkInputA
        else:
            mo = self.moveOutputB
            wo = self.winOutputB
            ni = self.networkInputB
        
        asyncs = []
        framesPerProc = int(len(frames) / threads)
        for i in range(threads):
            startIndex = framesPerProc * i
            endIndex = startIndex + framesPerProc
            if i == threads - 1:
                endIndex = len(frames)
            
            asyncs.append(pool.apply_async(fillTrainingSetPart0, args=(self.getGameInit(), self.getPlayerCount(), frames[startIndex:endIndex], startIndex, mo, wo, ni)))
        
        return asyncs

    def waitForBuffers(self, useBufferA, asyncs, numFrames):
        for asy in asyncs:
            asy.get()
        
        if useBufferA:
            mo = self.moveOutputA
            wo = self.winOutputA
            ni = self.networkInputA
        else:
            mo = self.moveOutputB
            wo = self.winOutputB
            ni = self.networkInputB
        
        assert torch.sum(ni.ne(ni)) == 0
        assert torch.sum(mo.ne(mo)) == 0
        assert torch.sum(wo.ne(wo)) == 0
        
        nIn = Variable(ni[:numFrames]).cuda()
        mT = Variable(mo[:numFrames]).cuda()
        wT = Variable(wo[:numFrames]).cuda()
        
        return nIn, mT, wT
    
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
        
        useBufferA = True
        
        asyncs = self.queueFillBuffers(pool, frames, useBufferA, threads)
        
        for e in range(self.epochs):
            pTimeStart = time.time()
            nIn, mT, wT = self.waitForBuffers(useBufferA, asyncs, len(frames))
            useBufferA = not useBufferA
            asyncs = self.queueFillBuffers(pool, frames, useBufferA, threads-1)

            print("Waited %f seconds for data!" % (time.time() - pTimeStart))
            
            mls = []
            wls = []
            
            bTimeStart = time.time()
            
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
                
            print("Completed Epoch %i with loss %f + %f in %f seconds" % (e, np.mean(mls), np.mean(wls), time.time() - bTimeStart))
        
        self.net.train(False)
        
        del nIn
        del mT
        del wT
        
        pool.terminate()
        
        torch.cuda.empty_cache()

