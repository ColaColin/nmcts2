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

class AbstractTorchLearner(AbstractLearner, metaclass=abc.ABCMeta):
    def __init__(self, framesBufferSize, batchSize, epochs, lr_schedule):
        assert framesBufferSize % batchSize == 0

        self.lr_schedule = lr_schedule
        self.framesBufferSize = framesBufferSize
        self.batchSize = batchSize
        self.epochs = epochs
        self.netInIsCached = False
        self.netInCache = None
    
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
        # TODO for larger input sizes this isn't such a good idea
        self.networkInput = torch.zeros((self.framesBufferSize,) + self.getNetInputShape())#.pin_memory()
        self.moveOutput = torch.zeros(self.framesBufferSize, self.getMoveCount()).pin_memory()
        self.winOutput = torch.zeros(self.framesBufferSize, self.getPlayerCount()).pin_memory()
        self.net = self.createNetwork()
        self.opt = self.createOptimizer(self.net)
        
        if file != None:
            self.net.load_state_dict(torch.load(file))
            print("Loaded state from " + file)
        
        self.net.cuda()
        self.net.train(False)
    
    def saveState(self, file):
        torch.save(self.net.state_dict(), file)
    
    """
    this has to be able to deal with None values in the batch!
    """
    def evaluate(self, batch):
        cdef int idx, bidx
        cdef int batchSize = len(batch)
        for idx in range(batchSize):
            b = batch[idx]
            if b is not None:
                state = b
                self.fillNetworkInput(state, self.networkInput , idx)

        if self.netInIsCached:#
            netIn = self.netInCache
        else:
            netIn = Variable(torch.zeros((self.batchSize, ) + self.getNetInputShape())).cuda()
            self.netInCache = netIn
            self.netInIsCached = True
        
        #TODO systematically analyze all interaction with gpu memory and apply new findings
        netIn[:len(batch)] = self.networkInput[:len(batch)]
        
        moveP, winP = self.net(netIn)
        
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
                    w.append(winP.data[bidx, state.mapPlayerIndexToTurnRel(pid)])
                
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
                
#             print(frame[0], "=>")
#             print(self.moveOutput[fidx], self.winOutput[fidx])
#             print(self.networkInput[fidx])
    
    def learnFromFrames(self, frames, iteration, dbg=False, reAugmentEvery=1):
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
        
        for e in range(self.epochs):
            
            print("Preparing example data...")
            
            if e % reAugmentEvery == 0:
                print("Filling with augmented data")
                self.fillTrainingSet(frames)
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
                self.opt.zero_grad()
                
                x = nIn[bi*self.batchSize : (bi+1) * self.batchSize]
                yM = mT[bi*self.batchSize : (bi+1) * self.batchSize]
                yW = wT[bi*self.batchSize : (bi+1) * self.batchSize] 
                
                if dbg:
                    print(x, yM, yW)
                
                mO, wO = self.net(x)
                
                mLoss = -torch.sum(mO * yM) / self.batchSize
                wLoss = -torch.sum(wO * yW) / self.batchSize
                
                loss = mLoss + wLoss
                loss.backward()
                
                # TODO maybe use some gradient clipping to be save?
                
                self.opt.step()
                
                mls.append(mLoss.data[0])
                wls.append(wLoss.data[0])
                
            print("Completed Epoch %i with loss %f + %f" % (e, np.mean(mls), np.mean(wls)))
        
        self.net.train(False)
        
        del nIn
        del mT
        del wT

