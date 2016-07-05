# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:15:47 2016
Description:
@author: Gurunath Reddy M
"""

import sys, csv, os
from essentia import *
from essentia.standard import *
from pylab import *
from numpy import *
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
#import stft as STFT
import _savitzky_golay as savfilt
from scipy.signal import medfilt


def musicVocaliNonVocalic(audio, hopSize=128, frameSize=2048, sampleRate=44100):
    #filename = '../segwav/3harmonicComp.wav'
    #hopSize = 128
    #frameSize = 2048
    #sampleRate = 44100
    #guessUnvoiced = True
    
    run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
    run_spectrum = Spectrum(size=frameSize * 4)
    run_spectral_peaks = SpectralPeaks(minFrequency=50,
                                   maxFrequency=10000,
                                   maxPeaks=100,
                                   sampleRate=sampleRate,
                                   magnitudeThreshold=0,
                                   orderBy="magnitude")
    run_pitch_salience_function = PitchSalienceFunction(magnitudeThreshold=60)
    run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks(minFrequency=90, maxFrequency=800)
#    run_pitch_contours = PitchContours(hopSize=hopSize, peakFrameThreshold=0.7)
#    run_pitch_contours_melody = PitchContoursMelody(guessUnvoiced=guessUnvoiced,
#                                                hopSize=hopSize)
    pool = Pool();
    #audio = MonoLoader(filename = filename)()
    #audio = EqualLoudness()(audio)
    
    for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        frame = run_windowing(frame)
        spectrum = run_spectrum(frame)
        specGram = pool.add('allframe_spectrum', spectrum);    
        peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
        salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
        salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)
        salSum = np.sum(np.power(salience_peaks_saliences, 2))
        pool.add('salienceSum', salSum)
        
    specGram = pool['allframe_spectrum']
    totalSalienceEnrg = pool['salienceSum']     
    totalSalienceEnrg = totalSalienceEnrg/np.max(totalSalienceEnrg)
    timeAxis = ((hopSize) * np.arange(np.size(totalSalienceEnrg)))/float(sampleRate)
    audioTime = np.arange(np.size(audio))/float(sampleRate)
    
    #plt.figure()
    #plt.plot(timeAxis, totalSalienceEnrg)    
    #plt.plot(audioTime, audio)
    
    totalSalienceEnrg = medfilt(totalSalienceEnrg, 3)
    totalSalienceEnrg = savfilt.savgol_filter(totalSalienceEnrg, 31, 3)     
    #plt.plot(totalSalienceEnrg)
    
    # Divide the signal into vocal and non-vocal regions
    delta = 0.9
    meanSalience = np.mean(totalSalienceEnrg)
    stdSalience = np.std(totalSalienceEnrg) * delta
    thresh = meanSalience - stdSalience    # Threshold is decided based on the mean and standard deviation of the SOE
    if (thresh < 0):                        # Check if threshold is negative
        while(meanSalience < stdSalience):
            delta = delta - 0.1
            stdSalience = np.std(totalSalienceEnrg) * delta 
        thresh = meanSalience - stdSalience
    
    tempVocal = (totalSalienceEnrg > thresh)
    vocalLoc = np.zeros(tempVocal.size)
    vocalLoc[tempVocal] = 1 
    
    # Each vocal region must be atleast 90ms
    for i in range(1, tempVocal.size-2):        # 
        if ((vocalLoc[i-1] == 1) and (vocalLoc[i] == 0) and (vocalLoc[i+1] == 0) and (vocalLoc[i+2] == 1)): # Two consequitive non-vocals b/w vocals
            vocalLoc[i] = 1
            vocalLoc[i+1] = 1
        elif((vocalLoc[i-1] == 1) and (vocalLoc[i == 0]) and (vocalLoc[i+1] == 1)):     # One non-vocal b/w two vocals
            vocalLoc[i] = 1
        elif((vocalLoc[i-1] == 0) and (vocalLoc[i] == 1) and (vocalLoc[i+1] == 0)):     # Vocal b/w non-vocals
            vocalLoc[i] = 0          
        
    # Finding begin and end of vocal regions
    diffVocalLoc = np.diff(vocalLoc)
    if (vocalLoc[0] == 1):                              # If the vocals begins from first frame itself then difference function ignores
        vocalBeg = np.where(diffVocalLoc == 1.0)[0]     # Positive difference corresponds to vocal begins
        vocalEnd = np.where(diffVocalLoc == -1.0)[0]    # Negative difference corresponds to vocal end index
        vocalBeg = np.append(0, vocalBeg)               # Vocals begins right from first frame include that index
        if (vocalLoc[-1] == 1):                         # Vocals continues till the end, then considere end of the signal as end of vocal regions
            vocalEnd = np.append(vocalEnd, np.size(vocalLoc))
    else:
        vocalBeg = np.where(diffVocalLoc == 1)[0]      # This case is the non-vocals followed by vocals
        vocalEnd = np.where(diffVocalLoc == -1)[0]
        if (vocalLoc[-1] == 1):
            vocalEnd = np.append(vocalEnd, np.size(vocalLoc))
    
    if (np.size(vocalBeg) != np.size(vocalEnd)):
        raise ValueError('The number of vocal begin indicies must be equal to the number of end indicies')
    
    # Saving a copy of vocal beg and end frame indicies
    vocalBegFrameIndex = np.copy(vocalBeg)
    vocalEndFrameIndex = np.copy(vocalEnd)
    
    # Converting vocal beg and end index values to time
    vocalBeg = (vocalBeg * hopSize)/float(sampleRate)
    vocalEnd = (vocalEnd * hopSize)/float(sampleRate)
    
    # Remove all those vocals which are less than 100ms
    vocalDiff = vocalEnd - vocalBeg
    tempVocalIndx = np.argwhere(vocalDiff > 0.1)  # 0.1 = 100ms
    vocalBeg = vocalBeg[tempVocalIndx]
    vocalEnd = vocalEnd[tempVocalIndx]
        
    # This is working fine  
    #plt.figure()
    #plt.stem(timeAxis, medFiltdZFFSEnrg/np.max(medFiltdZFFSEnrg), 'r')
    
    tempVocalUnitStep = np.ones(np.size(vocalEnd)) * 0.4      # Vocal region markers
    
    thresholdCurve = np.ones(np.size(totalSalienceEnrg)) * thresh
    
    plt.figure()
    plt.plot(audioTime, audio)
    plt.plot(timeAxis, totalSalienceEnrg)
    plt.plot(timeAxis, thresholdCurve, 'm')    
    plt.stem(vocalBeg, tempVocalUnitStep, 'g')
    plt.stem(vocalEnd, tempVocalUnitStep, 'r')
    
    return specGram, vocalBeg, vocalEnd, totalSalienceEnrg    
    #    
    #    pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
    #    pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)
    #
    #contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
    #        pool['allframes_salience_peaks_bins'],
    #        pool['allframes_salience_peaks_saliences'])
    #pitch, confidence = run_pitch_contours_melody(contours_bins,
    #                                              contours_saliences,
    #                                              contours_start_times,
    #                                              duration)
    #
    #figure(1, figsize=(9, 6))
    #
    #mX, pX = STFT.stftAnal(audio, sampleRate, hamming(frameSize), frameSize, hopSize)
    #maxplotfreq = 3000.0
    #numFrames = int(mX[:,0].size)
    #frmTime = hopSize*arange(numFrames)/float(sampleRate)                             
    #binFreq = sampleRate*arange(frameSize*maxplotfreq/sampleRate)/frameSize                       
    #plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:frameSize*maxplotfreq/sampleRate+1]))
    #plt.autoscale(tight=True)
    #
    #offset = .5 * frameSize/sampleRate
    #for i in range(len(contours_bins)):
    #  time = contours_start_times[i] - offset + hopSize*arange(size(contours_bins[i]))/float(sampleRate)
    #  contours_freq = 55.0 * pow(2, array(contours_bins[i]) * 10 / 1200.0)
    #  plot(time,contours_freq, color='k', linewidth = 2)
    #
    #plt.title('mX + F0 trajectories (carnatic.wav)')
    #tight_layout()
    #savefig('predominantmelody.png')
    #show()
