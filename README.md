# Melody-Extraction
A saliency based method for extracting the melody of the vocals in the polyphonic music signal. The first step in the proposed method consists of pre-filtering the polyphonic signal to emphasise the vocal frequency content in the composite signal. The pre-filtered signal is further processed to suppress the non-pitched percussion source by exploiting its wideband spectral characteristics. A novel weighted harmonic summation based frequency saliency computation method is proposed for the detected harmonic partials. Before saliency computation, harmonic partials from the percussion suppressed spectrum is obtained by the similarity based sinusoidal detection method. Further, the parameters (frequency and amplitude) of the detected partials are corrected by parabolic interpolation. The potential octave errors due to spurious peaks at multiple or sub-multiple of the true melody F0 is suppressed by introducing harmonic partial energy subtraction and Fourier similarity measure as refinement methods. The vocal and non-vocal regions in the mixture signal is obtained by thresholding the harmonic partial energy of the percussion suppressed music signal.

Folder "Saliency_Based_Melody_Extraction" contains all relevant source code.

"mainDominHarmSeries_V1.py" is the main Melody extraction script. It takes the Music excrept and the Melody extracted from Melodia (http://mtg.upf.edu/technologies/melodia) as input files.

Music excrepts (.wav files) and the melody contours (.txt files) extracted from Melodia are provided in "segwav" folder. 

Usage: python mainDominHarmSeries_V1.py
