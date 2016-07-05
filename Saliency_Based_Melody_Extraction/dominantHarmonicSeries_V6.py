# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:09:07 2016
Description: Dominant predicted harmonic series detection
@author: Gurunath Reddy M

"""
#import os, sys
import numpy as np
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '/home/gurunath/coursera/audio_signal_processing/sms-tools-master/software/models/utilFunctions_C'))
#import utilFunctions_C as UF_C
import peakDetectCorrect as pdc
import matplotlib.pyplot as plt
import FourierSimilarityForHarmonicMeasure as FourierSim
from scipy.signal import medfilt
#import waveio as io
#import dftStft as stft
#from scipy.signal import get_window
#import plotSpect as pltspct
#import onsetSpectDiff as onspdiff
from essentia import *
from essentia.standard import *
import essentiaSpecGram as essSpec

def resonanceFreqExtract(x, fs, M=2756, N=4096, H=256):
    #inputFile = '../segwav/3harmonicComp.wav'
    #fs = 44100
    #audio = MonoLoader(filename = inputFile)()  # Load audio file
    #x = np.copy(audio)                          # Make a copy of signal smaples to convert data type
    #x = np.array(x, np.float64)                 # Convert the samples to matlab double data type
    #x = x/(1.01*np.max(np.abs(x)));             # Normalize sample values
    #x = x - np.mean(x)                          # Perform mean subtraction to remove DC bias                
    lenX = x.size                               # Length of the signal in samples
    timeAxis = np.arange(lenX)/float(fs)        # Music signal time axis
    
    ## FFT parameter setting
    #M = 2756  # 1024
    #N = 4096  # 2048  
    #H = 256 # Corrresponds to 5ms of hop size
    #window = 'hamming'
    #w = get_window(window, M)
    
    mx = essSpec.essentiaSpect(x, H, M, fs)
    N = M + 2*M
    
    frameTime = 0.0         # Frame beginning time
    frameNumber = int(np.floor(frameTime * float(fs)/H))
    endTime = lenX/fs       # Frame end time
    frameEnd = int(np.floor(endTime * float(fs)/H))
    
    #f0t = 0 # TODO: Remove this in future
    
    f0 = np.array([])
    f0Twm = np.array([])
    f0MaxEnrg = np.array([])
    origF0MaxEnrg = np.array([])
    totalSaliency = np.array([])    # Save the energy of the dominant harmonic series
    
    for k in range(frameNumber, frameEnd):
        magFrame = mx[k, :]
        mxdbFrame = 20 * np.log10(magFrame)
        mX = mxdbFrame
        t = -70.0
        #        pX = stftPx[i, :]
        ploc = pdc.peakDetection(mX, t)              # detect peak locations   
        iploc, ipmag = pdc.peakInterp(mX, ploc)      # refine peak values
        ipfreq = fs * iploc/N                        # convert locations to Hz
        f0stable = 0.0
        #f0t = f0Twm(ipfreq, ipmag, 5, 50, 1000, f0stable)  # find f0
        
        freqAxis = np.arange(np.size(magFrame)) * float(fs)/N
        
    #    plt.plot(freqAxis, magFrame)
        #plt.stem(f0t, 0.05)
        
        minf0 = 50
        maxf0 = 1000
        pfreq = ipfreq
        pmag = ipmag
        
        f0c = np.argwhere((pfreq>minf0) & (pfreq<maxf0))[:,0]   # use only peaks within given range
        if (f0c.size == 0):                                     # return 0 if no peaks within range
            f0MaxEnrg = np.append(f0MaxEnrg, 0)                 # If there are no F0 candidate peaks then make F0 of that frame 0Hz (i.e., we have peaks above desired range)
            continue                                            # Go to next magnitude frame
            #raise ValueError('No candiate F0')
        f0cf = pfreq[f0c]                                       # frequencies of peak candidates
        f0cm = pmag[f0c]                                        # magnitude of peak candidates
        
        # Upto here it is clear. We are finding candidate frequencies within 50-4000Hz  
        
        #    plt.stem(f0cf, 10**(f0cm/20.0), 'r')
        
        f0t = 0 # TODO: Remove this in future
        
        if f0t>0:                                        # if stable f0 in previous frame 
            shortlist = np.argwhere(np.abs(f0cf-f0t)<f0t/2.0)[:,0]   # If already an F0 is detected in the previous frame, then the present F0 is most probabily within  f0t/2 - f0t < present F0 < f0t/2 + f0t  
            maxc = np.argmax(f0cm)          # Maximum amplitude peak in the measured peaks
            maxcfd = f0cf[maxc]%f0t         # Peak frequency corresponding to the max amplitude in the candidate f0 % previous f0 candidate
            print maxcfd
            if maxcfd > f0t/2:
                maxcfd = f0t - maxcfd
            if (maxc not in shortlist) and (maxcfd>(f0t/4)): # or the maximum magnitude peak is not a harmonic
                shortlist = np.append(maxc, shortlist) 
            f0cf = f0cf[shortlist]                         # frequencies of candidates                     
        
        #plt.figure()
        #plt.stem(f0cf, 10**(f0cm[shortlist]/20.0), 'g')
        
        # Understand this part and modify to include the weight of the dominant harmonic series
        
        #for k in range(1, 10):
        harmonicIndex = 1 # TODO: remove in future
        f0c = f0cf
        p = 0.5                                          # weighting by frequency value
        q = 1.4                                          # weighting related to magnitude of peaks
        r = 0.5                                          # scaling related to magnitude of peaks
        rho = 0.33                                       # weighting of MP error
        Amax = max(pmag)                                 # maximum peak magnitude
        maxnpeaks = 10                                   # maximum number of peaks used
        harmonic = np.matrix(f0c)
        ErrorPM = np.zeros(harmonic.size)                # initialize PM errors
        ErrorPMOrig = np.zeros(harmonic.size)                # initialize PM errors
        MaxNPM = min(maxnpeaks, pfreq.size)
        tempError = np.zeros(MaxNPM)
        simiMeasur = np.zeros(harmonic.size)
        predHarmEnrg = np.zeros(harmonic.size)
        relativeEnrg = np.zeros(harmonic.size)
        EuclidDist = np.zeros(harmonic.size)
        msrdharEnrg = np.zeros(harmonic.size)
        
        for i in range(0,  MaxNPM):                             # predicted to measured mismatch error
            difmatrixPM = harmonic.T * np.ones(pfreq.size)      # Each row is a single frequency corresponding to the candidate frequency
            difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)  # Subtract each candidate frequency with the peak frequencies
            FreqDistance = np.amin(difmatrixPM, axis=1)         # Minimum frequency distance between the candidate frequencies and the measured frequencies
            peakloc = np.argmin(difmatrixPM, axis=1)            # Peak locations (vector) corresponding to the min. frequency distance         
            Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p)) # Relative frequency difference
            PeakMag = pmag[peakloc]                             # Peak magnitudes corresponding to the min. frequency distance
            MagFactor = 10**((PeakMag-Amax)/20.0)               # Relative magnitudes wrt Max. peak magnitude
            ErrorPMOrig = ErrorPMOrig + (Ponddif+ MagFactor * (q*Ponddif-r)).T                      # Original TWM
    # +++++++++++++++++++++  Modifification part +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            newdifmatrixPM = np.power(abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq), 2)  # Subtract each candidate frequency with the peak frequencies
            newFreqDistance = np.amin(newdifmatrixPM, axis=1)         # Minimum frequency distance between the candidate frequencies and the measured frequencies
            newP = 2
            newPonddif = np.array(newFreqDistance) * (np.array(harmonic.T)**(-p)) # Relative frequency difference
    
            measuredMag =  1 + 10**(PeakMag/20.0)    # Nearest measured peaks corresponding to the ideal harmonics    
            locPredF0 = np.array(np.floor(harmonic * N/fs), dtype=int)   # Gives the bin number of predicted harmonics. Useful for getting the magnitude 
            magPredF0 = 1 + magFrame[locPredF0]   # Magnitude values of the predicted F0 derived from the magnitude spectrum
            
            simiMeasur = simiMeasur + measuredMag.T * magPredF0
            predHarmEnrg = predHarmEnrg + np.power(magPredF0, 2)
            msrdharEnrg = msrdharEnrg + np.power(measuredMag, 2).T
    #        relativeEnrg = relativeEnrg + 1 - simiMeasur/predHarmEnrg
            relativeEnrg = relativeEnrg + (1 - (simiMeasur/predHarmEnrg))
            #print np.argmin(relativeEnrg)
            EuclidDist = EuclidDist + abs(measuredMag.T - magPredF0)    
        #    print EuclidDist        
    #        ErrorPM = ErrorPM + (Ponddif + EuclidDist.T * relativeEnrg.T * (q*Ponddif-r)).T        # Modified TWM        
    #        ErrorPM = ErrorPM + (newPonddif + EuclidDist.T * relativeEnrg * (q * newPonddif - r)).T       # Modified TWM        
            ErrorPM = ErrorPM + (newPonddif)
            harmonic = harmonic+f0c     # Next higher harmonic (These are our predicted harmonics)
    #    print f0cf[np.argmax(predHarmEnrg)]
            
        # The dominant energy may corresponds to the lower octave
        # TODO: Check why it is not working
        # TODO: This won't work unless exact sinusoids are detected
        if(predHarmEnrg.size > np.argmax(predHarmEnrg)):                        # Confirming that there is next harmonic series
            tempDominantHarm = f0cf[np.argmax(predHarmEnrg)]                    # Fundamental Frequncy index of dominat harmonic series
            templocDominantF0 = np.int(np.floor(tempDominantHarm * N/fs))       # Gives the bin number of predicted harmonics. Useful for getting the magnitude 
            tempHarm = templocDominantF0 * np.arange(1, MaxNPM)                 # Taking only even harmonics
            enrgDomiHar =  np.sum(np.power((1 + magFrame[tempHarm]), 2))
            
            # Finding the energy of harmonic series next to the hypothesised dominant harmonic series
            freqBinsPredF0 = np.array(np.floor(f0c * N/fs), dtype=int)
            nextOctave = 2*templocDominantF0
            # TODO: Check how to handle this case (24-01-2016)        
            nearestHigherOctave = np.argwhere((freqBinsPredF0 >= nextOctave-5) & (freqBinsPredF0 <= nextOctave+5))[:, 0]      
            if(nearestHigherOctave.size > 0):                                                       # We have detected more than one higher octave index
                nearDist = np.abs(nextOctave - freqBinsPredF0[nearestHigherOctave])
                nextHarLoc = f0cf[nearestHigherOctave[np.argmin(nearDist)]]
            elif((nearestHigherOctave.size > 0) and (nearestHigherOctave in freqBinsPredF0)):       # Check whether the next octave 
                nextHarLoc = f0cf[nearestHigherOctave]                                          # Fundamental Frequncy index of next harmonic series
            else:
                nextHarLoc = templocDominantF0                                                      # If it is anything else please keep the hypothesis dominant F0 as the actual F0
            f0NextHarLoc = np.array(np.floor(nextHarLoc * N/fs), dtype=int)                         # Gives the bin number of predicted harmonics. Useful for getting the magnitude 
            nextHarSeries = f0NextHarLoc * np.arange(1, MaxNPM)                                     # Taking only even harmonics
            enrgNextHar =  np.sum(np.power((1 + magFrame[nextHarSeries]), 2))                       # Finding the energy
            
            # Check how many frequency bins are overlapping
            nextHarSeriesMat  = np.matrix(nextHarSeries).T * np.ones(tempHarm.size)
            difmatrix = abs(nextHarSeriesMat - np.ones((nextHarSeries.size, 1))*tempHarm) 
            tempminDist = np.argwhere((difmatrix <= 5))                                             # Matrix coordinates of the position of zeros (i.e., frequencies which are exactly matching)
            minDistIndx = tempminDist[:, 0, 1]                                                      # Somehow it became a three dim matrix hence this circus
    #       If there are more than one non-overlapping harmonics (i.e., we are finding next harmonic series)
            if(minDistIndx.size > 0):
                enrgOverlapSeries = np.sum(np.power((1+magFrame[minDistIndx]), 2))
                enrgNonOvelpSeries = enrgDomiHar - enrgOverlapSeries            
                if(enrgNextHar > enrgNonOvelpSeries):
                    # Here we may falsly hypothesise next ocatve as dominant harmonic. Hence finding the Fourier similarity between the magnitude spectrum and the basis corresponding to the bin frequency
                    dominantHarCoeff = FourierSim.FourierSimilarityOfHarmonics(magFrame, np.array([templocDominantF0, nextOctave]))
                    if(dominantHarCoeff == 0):      # dominantHarCoeff = 0 corresponds to tempDominantHarm, 1 = nextHarLoc
                        f0MaxEnrg = np.append(f0MaxEnrg, tempDominantHarm) # Next harmonic is the dominat harmonic
                        #totalSaliency = np.append(totalSaliency, enrgDomiHar)
                    else:
                        f0MaxEnrg = np.append(f0MaxEnrg, nextHarLoc)
                        #totalSaliency = np.append(totalSaliency, enrgNextHar)
            else:
                f0MaxEnrg = np.append(f0MaxEnrg, tempDominantHarm)
        
        origF0MaxEnrg = np.append(origF0MaxEnrg, f0cf[np.argmax(predHarmEnrg)])
        Error = (ErrorPM[0]/MaxNPM)                                             # total error
        f0index = np.argmin(Error)                                              # get the smallest error
        f0 = np.append(f0, f0cf[f0index])                                       # f0 with the smallest error
    
    medFiltF0 = medfilt(f0MaxEnrg, 9)
    return medFiltF0
#    totalFrames = np.arange(frameNumber, frameEnd)
#    framesTime = (totalFrames * H)/float(fs)
    
    #melodiaF0File = '../segwav/7.txt'
    #melodiaF0 = np.loadtxt(melodiaF0File)
    #
    #plt.figure()
    #plt.stem(framesTime, medFiltF0, 'g')
    ##plt.plot(melodiaF0[:, 0], melodiaF0[:, 1], 'b')
    #plt.ylim([0, 500])
    #plt.figure()
    #plt.stem(framesTime, totalSaliency/np.max(totalSaliency))
    #plt.title('Saliency of the Harmonic source')
    
    #plt.plot(medFiltF0)
    #meanF0contour = np.mean(medFiltF0)
    #stdDevF0Contour = np.std(medFiltF0)
    #plt.plot(np.ones(medFiltF0.size) * meanF0contour)
    #plt.plot((np.ones(medFiltF0.size) * meanF0contour) + stdDevF0Contour, 'r')
    
    #plt.stem(framesTime, origF0MaxEnrg, 'r')
    #plt.plot(framesTime, f0Twm, 'r')
