import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
t=np.linspace(0,np.pi,100)

class modulation:
    
    def car(self,phase):
        c=np.cos((2*np.pi*t)-phase)
        e=np.linalg.norm(c)
        nc=c/e
        return nc

    #BPSK of bit stream
    def bpsk(self,a,nc):
        l=np.array([])
        for bit in a:
            if bit==0:
                l=np.concatenate([l,-nc],axis=0)
            else:
                l=np.concatenate([l,nc],axis=0)
        return l

    #Trasmission through Channel
    def awgn(self,k,sigma):
        n=np.random.normal(0,sigma,k.size)
        k=k+n 
        return k

    #demodulation
    def demod(self,l,nc):  
        z=len(l)
        y=len(nc)
        h=[]
        con=np.split(l,z/y)
        for ar in con:
            h.append(np.sum(nc*ar))
        h=np.array(h)
        return h 