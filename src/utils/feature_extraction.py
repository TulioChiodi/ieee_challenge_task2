from scipy.signal import stft
import numpy as np



def spectrum_fast(x,
                  nperseg=512, 
                  noverlap=128, 
                  window='hamming', 
                  cut_dc=True,
                  output_phase=True, 
                  cut_last_timeframe=True):
    '''
    Compute magnitude spectra from monophonic signal
    '''

    f, t, seg_stft = stft(x,
                        window=window,
                        nperseg=nperseg,
                        noverlap=noverlap)

    output = np.abs(seg_stft)

    if output_phase:
        phase = np.angle(seg_stft)
        output = np.concatenate((output,phase), axis=-3)

    if cut_dc:
        output = output[:,1:,:]

    if cut_last_timeframe:
        output = output[:,:,:-1]


    return output