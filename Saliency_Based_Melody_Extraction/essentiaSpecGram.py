# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:47:32 2015
Description:
Inputs:
Outputs:
@author: Gurunath Reddy M

"""

from essentia import *
from essentia.standard import *
import numpy as np

def essentiaSpect(audio, hopSize, frameSize, sampleRate):
    run_windowing = Windowing(type='hann', zeroPadding=2*frameSize) # Hann window with x4 zero padding
    run_spectrum = Spectrum(size=frameSize * 4)
    
    pool = Pool();
    
    for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        frame = run_windowing(frame)
        spectrum = run_spectrum(frame)
        specGram = pool.add('allframe_spectrum', spectrum);
        
    specGram = pool['allframe_spectrum']
    
    return specGram
