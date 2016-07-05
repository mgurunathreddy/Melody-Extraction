# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:12:44 2016
Description:
@author: Gurunath Reddy M
"""
import numpy as np
import matplotlib.pyplot as plt
#plt.stem(lowerOctHarm, np.ones(lowerOctHarm.size)*0.5, 'c')
#plt.stem(tempHarm, np.ones(tempHarm.size)*0.2, 'r')
#plt.stem(nextHarSeries, np.ones(nextHarSeries.size)*0.1, 'g')

def FourierSimilarityOfHarmonics(magFrame, f):
    '''
        Measures the Fourier similarity between the Fourier magnitude spectrum and the hypothesied harmonic series fundamental
        Input:  magFrame -> Fourier magnitude spectrum
                f        -> harmonic base frequencies (in bins)
        Output: Frequency Bin number corresponding to true harmonic series (0  = predicted or 1 = next octave) 
    '''
    # Take peaks in the magnitude spectrogram and then compute FT at the peaks
    magFrame = magFrame/np.max(magFrame)    # Normalized magnitude spectrum
    magFrame = magFrame[:300]               # Considering frequencies upto 1600 Hz
    lengthFFTFrame = np.size(magFrame)      # Length of the magnitude frame
    #f = np.array([lowerOctHarm[0], tempHarm[0], nextHarSeries[0]])  # Trial bin frequencies whose similarites needs to be tested 
    freq = 1./f     # Convering bins to frequencies
    #simWindow = np.hamming(lengthFFTFrame)
    #windowedMagFFT = (1 + magFrame) * simWindow
    
    T = np.arange(lengthFFTFrame)   # Generating time axis to the length of magnitude frame
    twoPiT = 2 * np.pi * T
    
    X = np.zeros(np.size(f), dtype='complex')
    
    for f0 in range(np.size(f)):
        twoPiFT = freq[f0] * twoPiT
        cosine = np.cos(twoPiFT)
        sine = np.sin(twoPiFT)
        co = np.sum(magFrame * cosine)
        si = np.sum(magFrame * sine) 
        x = (co + 1j * si)
        X[f0] = x
       
    #maxSimilarityFreq = f[np.argmax(abs(X))]
    maxSimilarityFreq = np.argmax(abs(X))
    #print abs(X)
    
    return maxSimilarityFreq    # Return the bin corresponding to maximum similarity measure            
if __name__ == "__main__":
    dominatFourierCoefficent = FourierSimilarityOfHarmonics(magFrame, f)
#    plt.figure()
#    plt.plot(cosine)
#    plt.plot(sine)
#    plt.plot(magFrame)    