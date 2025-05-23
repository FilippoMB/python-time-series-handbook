import numpy as np
import matplotlib.pyplot as plt

def fourierPrediction(y, n_predict, n_harm = 5):
    """
    Predict the future values of a time series using the Fourier series.

    Parameters
    ----------
    y : array-like
        The time series to predict.
    n_predict : int
        The number of future values to predict.
    n_harm : int
        The number of harmonics to use in the Fourier series.

    Returns
    -------
    out : array-like
        The predicted values of the time series.
    """
    n = y.size                         # length of the time series
    t = np.arange(0, n)                # time vector
    p = np.polyfit(t, y, 1)            # find linear trend in x
    y_notrend = y - p[0] * t - p[1]    # detrended x
    y_freqdom = np.fft.fft(y_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    
    # Sort indexes by largest frequency components
    indexes = np.argsort(np.absolute(y_freqdom))[::-1]

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        amp = np.absolute(y_freqdom[i]) / n   # amplitude
        phase = np.angle(y_freqdom[i])        # phase
        restored_sig += amp * np.cos(2 * np.pi * f[i] * t + phase)

    out = restored_sig + p[0] * t + p[1] # add back the trend
    return out


def annotated_sin_plot():
    """
    Plot a sine wave with a phase shift and annotate it.
    """
    A = 1
    f = 1  
    T = 1 / f
    omega = 2 * np.pi * f  
    phi = 0.5
    t = np.linspace(0, T, 1000)
    y = A * np.sin(omega * t)
    y_phi =  A * np.sin(omega * t + phi)
    plt.figure(figsize=(8, 5))
    plt.plot(t, y)
    arrow_idx = len(t) // 2 - 20
    t_arrow = t[arrow_idx]
    y_arrow = y[arrow_idx]
    plt.plot(t, y_phi, color='tab:red', linestyle='--')
    plt.annotate('', xy=(t_arrow-phi/(2*np.pi), y_arrow), xytext=(t_arrow, y_arrow),
                 arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
    plt.text(t_arrow-phi/(3*np.pi), y_arrow+0.1, r'$\psi$', va='center', color="k")
    plt.xlim(-0.1, T+0.1)
    plt.ylim(-A-0.2, A+0.2)
    xticks = [0, 1/4, 1/2, 3/4, 1]
    xtick_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    plt.xticks(xticks, xtick_labels)
    plt.xlabel('Radians')
    ax2 = plt.gca().twiny()  # Create a twin Axes sharing the yaxis
    ax2.set_xlim(plt.xlim())  # Ensure the limits are the same
    ax2.set_xticks(xticks)  # Use the same x-ticks as ax1
    ax2.set_xticklabels(['0', '90', '180', '275', '360'])  # But with degree labels
    ax2.set_yticks([])  # Hide the y-axis ticks
    plt.xlim(-0.1, T+0.1)
    plt.xlabel('Degrees')
    plt.text(0.11, -0.1, 'time ($t$)', ha='right')
    plt.text(-0.03, A+0.02, 'A', ha='right')
    plt.text(-0.03, 0+0.02, '0', ha='right')
    plt.text(-0.03, -A+0.02, '-A', ha='right')
    plt.text(T+0.05, 0, r'$T = 1/f$', va='bottom', ha='right')
    plt.text(T / 2 - 0.38, -A + 0.5, 'f = frequency\nT = period\nA = amplitude', ha='center', va='top')
    plt.ylabel('Amplitude')
    plt.axhline(A, color='gray', linestyle='--', linewidth=1)
    plt.axhline(-A, color='gray', linestyle='--', linewidth=1)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()