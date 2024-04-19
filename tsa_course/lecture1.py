import numpy as np
from scipy.fft import fft

def fft_analysis(signal):
    """
    Perform a Fourier analysis on a time series.

    Parameters
    ----------
    signal : array-like
        The time series to analyze.

    Returns
    -------
    dominant_period : float
        The dominant period of the time series.
    positive_frequencies : array-like
        The positive frequencies.
    magnitudes : array-like
        The magnitudes of the positive frequencies.
    """
    
    # Linear detrending
    slope, intercept = np.polyfit(np.arange(len(signal)), signal, 1)
    trend = np.arange(len(signal)) * slope + intercept 
    detrended = signal - trend 
    
    fft_values = fft(detrended)
    frequencies = np.fft.fftfreq(len(fft_values))

    # Remove negative frequencies and sort
    positive_frequencies = frequencies[frequencies > 0]
    magnitudes = np.abs(fft_values)[frequencies > 0]

    # Identify dominant frequency
    dominant_frequency = positive_frequencies[np.argmax(magnitudes)]

    # Convert frequency to period (e.g., days, weeks, months, etc.)
    dominant_period = 1 / dominant_frequency
    
    return dominant_period, positive_frequencies, magnitudes