import numpy as np

class TimeSeries:
    # Time series satisfying the Poisson distribution, and sum is 1
    def __init__(self, queryNum) -> None:
        self.len = queryNum
        tSeq = np.random.poisson(lam = 1e6 / queryNum, size=queryNum)
        self.timeSeq = tSeq / np.sum(tSeq)
    def __call__(self, i):
        assert i < self.len
        return self.timeSeq[i]


class Info:
    def __init__(self) -> None:
        self.inferTime = []
        self.encodeTime = []
        self.decodeTime = []
        self.totalTime = 0
    
    def add_infertime(self,t):
        self.inferTime.append(t)
    
    def add_encodetime(self,t):
        self.encodeTime.append(t)
        
    def add_decodetime(self,t):
        self.decodeTime.append(t)
    
    def set_totaltime(self,t):
        self.totalTime = t
    
    def output(self):
        # assert(len(self.inferTime) == self.questNum)
        print("----------------------------------")
        print(len(self.inferTime))
        print("Median:")
        print(np.median(self.inferTime))
        print("Mean:")
        print(np.mean(self.inferTime))
        print("99th:")
        print(np.percentile(self.inferTime, 99))
        print("99.5th:")
        print(np.percentile(self.inferTime, 99.5))
        print("99.9th:")
        print(np.percentile(self.inferTime, 99.9))
        print("----------------------------------")
        print(len(self.encodeTime),len(self.decodeTime))
        print("Mean(encode):")
        print(np.mean(self.encodeTime))
        print("Mean(decode):")
        if len(self.decodeTime) == 0:
            print("no fail")
        else:
            print(np.mean(self.decodeTime))
        print("----------------------------------")
        print("total time:")
        print(self.totalTime)
        print("----------------------------------")

INFO = Info()

class ImageInfo:
    def __init__(self, id, fpath):
        self.id = id
        self.path = fpath
        self.byte = open(fpath, "rb").read()