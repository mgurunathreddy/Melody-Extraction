# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:51:20 2016
Description: Melody (sequence of F0 values) extraction from vocal polyphonic music signals.
Dependencies: Esentia   - audio feature extraction tool box (http://essentia.upf.edu/)
              SMS tools - (Installation is not required) Spectral modeling and sysnthsis tool box (https://github.com/MTG/sms-tools). Some of the functions are used with acknowledgement. No need to install explicitely   
              Numpy     - numerical python package
              Scipy     - scientific python package
              Matplotlib- python plotting package
Inputs: Music excrept smapled at 44.1KHz and the time frequency pairs of melody obtained from Melodia plug-in for comparision.
Outputs: Melody of the vocals
@author: Gurunath Reddy M

"""
#==============================================================================
# import waveio as io
# import dftStft as stft
# from scipy.signal import get_window
# import plotSpect as pltspct
# import onsetSpectDiff as onspdiff
# import essentiaSpecGram as essSpec
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from essentia import *
from essentia.standard import *

inputFile = '../segwav/3.wav'
melodiaF0File = '../segwav/3.txt'       # Melody contour obtained from the Melodia. Please comment if you don't have melody extracted from Melodia (http://mtg.upf.edu/technologies/melodia)
 
fs = 44100.0
audio = MonoLoader(filename = inputFile)()   # Load audio file
audio = EqualLoudness()(audio)               # Passing music signal through an equal loudness filter to emphasise vocal regions
#x = np.copy(audio)                          # Make a copy of signal smaples to convert data type
#x = np.array(x, np.float64)                 # Convert the samples to matlab double data type
#x = x/(1.01*np.max(np.abs(x)));             # Normalize sample values
#x = x - np.mean(x)                          # Perform mean subtraction to remove DC bias                
#lenX = x.size                               # Length of the signal in samples
#timeAxis = np.arange(lenX)/float(fs)        # Music signal time axis
window = 'hamming'

# Percussion suppression of the vocal polyphonic music signal
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#==============================================================================
# import percussionSuppress as percSupp
# audio = percSupp.percusionSeparation(audio, fs, window, M=2048, H=512, N=4096, inputFile='harmonic')
# audio = essentia.array(y)
#==============================================================================

# Vocal and Non-Vocal detecton
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
hopSize = 256
frameSize = 2048
import saliencyBasedVUV as vnv
specGram, vocalBeg, vocalEnd, totalSalienceEnrg = vnv.musicVocaliNonVocalic(audio, hopSize, frameSize, fs)

# Converting vocal beg and end index values to time
vocalBeg = (vocalBeg * fs) / hopSize            # Vocal begin in frame number
vocalEnd = (vocalEnd * fs) / hopSize            # Vocal end in frame number    

# Melody contour detection
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
M = 2756  # 1024
N = 4096  # 2048  
H = 256 # Corrresponds to 5ms of hop size    
import dominantHarmonicSeries_V6 as domintharmic
medFiltResonF0 =  domintharmic.resonanceFreqExtract(audio, fs, M, N, H)
timeMedFiltRes = (np.arange(np.size(medFiltResonF0)) * H) /float(fs)

plt.figure()    
plt.plot(timeMedFiltRes, medFiltResonF0, 'g')
melodiaF0 = np.loadtxt(melodiaF0File)
plt.plot(melodiaF0[:, 0], melodiaF0[:, 1], 'r') # Shift the melody obtained by Melodia plugin-up by 50Hz for visualization
plt.ylim([0, 500])
plt.xlim([0, np.max(timeMedFiltRes)])
plt.title('Melody without Vocal and Non-Vocal Detection (P = proposed, M = Melodia) ')
plt.xlabel('Time(s)')
plt.ylabel('Frequency(Hz)')
plt.legend(['P', 'M'])

vocalMelody = np.zeros(np.size(medFiltResonF0))
for i in range(np.size(vocalBeg)):
    begFrame = np.int(vocalBeg[i])
    endFrame = np.int(vocalEnd[i])
    vocalMelody[begFrame:endFrame] = medFiltResonF0[begFrame:endFrame]

plt.figure()    
plt.plot(timeMedFiltRes, vocalMelody, 'g')
#melodiaF0 = np.loadtxt(melodiaF0File)
plt.plot(melodiaF0[:, 0], melodiaF0[:, 1], 'r') # Shift the melody obtained by Melodia plugin-up by 50Hz for visualization
plt.ylim([0, 500])
plt.xlim([0, np.max(timeMedFiltRes)])
plt.title('Melody with Vocal and Non-Vocal Detection (P = proposed, M = Melodia) ')
plt.xlabel('Time(s)')
plt.ylabel('Frequency(Hz)')
plt.legend(['P', 'M'])
