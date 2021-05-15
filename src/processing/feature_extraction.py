import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# ===================================================================================
## You must load every audio with librosa.load(audio_path, sr=32000, mono=False),
## this is going to be "audio" (the input for the functions below). I'LL FIX THIS!
# ===================================================================================

def get_stft(audio, fs=32000, n_fft=2048, hop_length=512):
    """
    Compute stft from an ambisonics input (4 channels).
    Input:
        - input: ambisonics audio [ch, total_samples]
        - fs: sampling rate
        - n_fft: total number of samples per frame
        - hop_length: hop length
    Ouput:
        - STFT for each channel and frequency bins.
        [number_of_frequency_bins, number_of_time_bins, n_channels]
    """
    # STFT vector final shape
    frame_bins_size = (audio.shape[1]) // hop_length + 1  # Number of time frames
    freq_bins_size = n_fft // 2 + 1  # Number of frequency bins
    n_ch = audio.shape[0]  # Number of channels

    audio_stft = np.zeros((n_ch, freq_bins_size, frame_bins_size), dtype=complex) # Zero vector with output shape

    # Compute stft for each channel
    for ch in range(n_ch):
    
        stft = librosa.stft(audio[ch, :], n_fft=n_fft, hop_length=hop_length)
        audio_stft[ch, :, :] = stft[:, :]
    
    freq_bins = np.arange(0, 1 + n_fft / 2) * fs / n_fft # Frequency bins vector
    
    return audio_stft, freq_bins


def get_mel_spectrogram(audio, fs=32000, n_fft=2048, hop_length=512, n_mel_bands=10):
    """
    Compute stft from an ambisonics input (4 channels).
    Input:
        - input: ambisonics audio [ch, total_samples]
        - fs: sampling rate
        - n_fft: total number of samples per frame
        - hop_length: hop length
    Ouput:
        - STFT for each channel and frequency bins.
    """
    frame_vec_size = (audio.shape[1]) // hop_length + 1 # Number of time frames
    n_channels = audio.shape[0] # Number of channels
    
    mel_spectrogram = np.zeros((n_channels, n_mel_bands, frame_vec_size), dtype='float32') # Zero vector with output shape
    
    for ch in range(n_channels):
        
        mel = librosa.feature.melspectrogram(audio[ch, :], sr=fs, n_fft=n_fft,
                                             hop_length=hop_length, n_mels=n_mel_bands)
        mel_spectrogram[ch, :, :] = mel[:, :]
    
    return mel_spectrogram

    
def get_logmel_spectrogram(audio, fs=32000, n_fft=2048, hop_length=512, n_mel_bands=10):
    """
    Compute stft from an ambisonics input (4 channels).
    Input:
        - input: ambisonics audio | [ch, total_samples]
        - fs: sampling rate
        - n_fft: total number of samples per frame
        - hop_length: hop length
    Ouput:
        - STFT for each channel and frequency bins.
    """
    logmel_spectrogram = get_mel_spectrogram(audio, fs, n_fft, hop_length, n_mel_bands)
        
    logmel_spectrogram = librosa.power_to_db(logmel_spectrogram)
    
    return logmel_spectrogram


def get_intensity_vector(audio_stft, frame_len, fs, n_mel_bands):
    """
    Compute intensity vector. Input is a four channel stft of the signals.
    """
    IVx = np.real(np.conj(audio_stft[0, :, :]) * audio_stft[1, :, :])
    IVy = np.real(np.conj(audio_stft[0, :, :]) * audio_stft[2, :, :])
    IVz = np.real(np.conj(audio_stft[0, :, :]) * audio_stft[3, :, :])
    
    # Mel filterbanks
    mel_weights = librosa.filters.mel(n_fft=frame_len, sr=fs, n_mels=n_mel_bands)
    
    # Menor número representável (?), é necessário porque se não ocorre uma divisão por zero na normalização.
    epsilon = 1e-8
    
    # Norm vector
    norm = np.sqrt(IVx**2 + IVy**2 + IVz**2) + epsilon
    
    # Intensity vector for each direction
    IVx_norm = np.matmul(mel_weights, (IVx / norm))
    IVy_norm = np.matmul(mel_weights, (IVy / norm))
    IVz_norm = np.matmul(mel_weights, (IVz / norm))
       
    intensity_vector = np.stack((IVx_norm, IVy_norm, IVz_norm), axis=0)
    
    return intensity_vector

def get_logmel_IV(audio, fs=32000, n_fft=2048, hop_length=512, n_mel_bands=100, frame_length=2049):
    
    audio_stft, freq_bins = get_stft(audio,fs,n_fft,hop_length)
    logmel_stft = get_logmel_spectrogram(audio,fs,n_fft, hop_length,n_mel_bands)
    int_vector = get_intensity_vector(audio_stft,frame_length,fs,n_mel_bands)
    
    return np.concatenate((int_vector,logmel_stft), axis=0)


def plot_spectrogram(audio_stft, ch=0, fs=32000, hop_length=512, y_axis='log'):
    """
    Plot a linear power spectrogram for the selected channel from a 4-channel stft input.
    Input: 
        - 4-channel stft.
    Output:
        - Linear power spectrogram plot.
    """
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(audio_stft[ch, :, :], sr=fs, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    
    plt.colorbar(format='%+2.f')
    
    
def plot_spectrogram_db(audio_stft, ch=0, fs=32000, hop_length=512, y_axis='log'):
    """
    Plot a log power spectogram for the selected channel from a 4-channel stft input.
    Input: 
        - Four-channel stft
    Output:
        - Log power spectrogram plot
    """
    stft = librosa.power_to_db(np.abs(audio_stft[ch, :, :])**2)
    
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(stft[:, :], sr=fs, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    
    plt.colorbar(format='%+2.f dB')
    
    
def plot_logmel_spectrogram(audio_logmel_spectra, ch=0, fs=32000, hop_length=512):
    """
    Plot a log mel spectrogram for the selected channel from a 4-channel logmel spectrogram input.
    Input: 
        - 4-channel logmel spectrogram.
    Output:
        - Log power spectrogram plot for selected channel.
    """
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(audio_logmel_spectra[ch, :, :],
                             x_axis='time',
                             y_axis='mel',
                             sr=fs,
                             hop_length=hop_length)
    plt.colorbar(format='%+2.f dB')
    plt.show()
