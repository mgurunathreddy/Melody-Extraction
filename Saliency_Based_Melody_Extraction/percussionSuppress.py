# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 11:23 2016
Description: Takes the input music signal, does the percussion suppression and then saves it to segwav folder a wave file
Inputs:
Outputs:
@author: Gurunath Reddy M

"""
import numpy as np
import stftAnalSynth as stftAnalSyn
import waveio as io
from scipy.signal import get_window
import matplotlib.pyplot as plt
import numpy as np

def percusionSeparation(x, fs, window, M=2048, H=512, N=4096, inputFile='harmonic'):
    # Compute the analysis window
    w = get_window(window, M)
    
    # compute the magnitude and phase spectrogram
    mX, pX = stftAnalSyn.stftAnal(x, fs, w, N, H)       # mX is a linear magnitude and pX is the phase
    S = np.transpose(mX)
    phase = np.transpose(pX)
    
    # +++++++++++++++++ Spectral distance measure along the frequency axis ++++++++++++++++++++++++++
    tempSpectDiffAlongFrequency = np.diff(S, axis=0)    # Spectral diff along frequency axis
    tempAbsSpectDiffAlongFreq = tempSpectDiffAlongFrequency * (tempSpectDiffAlongFrequency > 0)
    #tempAbsSpectDiffAlongFreq = np.abs(tempSpectDiffAlongFrequency)
    tempAbsSpectDiffAlongFreq = tempAbsSpectDiffAlongFreq ** 2
    distMeasAlongFreq = np.vstack((tempAbsSpectDiffAlongFreq[0, :], tempAbsSpectDiffAlongFreq))
    
    # Creating binary mask for the entire signal
    harmBinaryMask = (distMeasAlongFreq > (np.max(distMeasAlongFreq) * 0.01/100)).astype(float)   # (0.01 is the optimum) value Binary mask creation
    #percBinaryMask = 1 - harmBinaryMask
    
    # Retaining harmonic components in the complex spectrogram
    complxHarmonicSpcgrm =  (S * harmBinaryMask) * phase
    
    # Harmonic component extraction and synthesis
    absCompHarmSpect = abs(complxHarmonicSpcgrm)                                        # compute ansolute value of positive side
    absCompHarmSpect[absCompHarmSpect<np.finfo(float).eps] = np.finfo(float).eps        # if zeros add epsilon to handle log
    magHarmSpect = 20 * np.log10(absCompHarmSpect)                                      # magnitude spectrum of positive frequencies in dB         
    phaseHarmSpect = np.unwrap(np.angle(complxHarmonicSpcgrm))                          # unwrapped phase spectrum of positive frequencies
    magHarmSpecgrm = np.transpose(magHarmSpect)
    phaseHarmSpecgram = np.transpose(phaseHarmSpect)
    
    # perform the inverse stft
    yHarmonic = stftAnalSyn.stftSynth(magHarmSpecgrm, phaseHarmSpecgram, M, H)
    # Write to a wavefile 
    #io.wavwrite(yHarmonic, fs, inputFile[:-4] + 'harmonicComp.wav')
    return yHarmonic

if __name__ == "__main__":

    inputFile = '../segwav/2.wav'
    fs, x = io.wavread(inputFile)
    M = 2048
    N = 4096
    H = M/4
    #H = 132
    window = 'hamming'
    yHarmonic = percusionSeparation(x, fs, window, M, H, N, inputFile)   # Percussion supressed music signal        
    # Write to a wavefile 
    io.wavwrite(yHarmonic, fs, inputFile[:-4] + 'harmonicComp.wav')