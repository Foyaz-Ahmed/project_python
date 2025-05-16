import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file with original sample rate
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Extract MFCCs
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Average across time to get a fixed-size feature vector
    return mfcc_mean