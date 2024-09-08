# ------------------------------------------------------------------------------------------------
# This code contains the functions related to audio processing (record, compute spectrogram, ..)
# ------------------------------------------------------------------------------------------------


import random
import torch
import torchaudio
import pyaudio
import wave
from torchaudio import transforms


# general audio parameters
fs=16000
n_channels = 1
duration_recording_dataset = 3 # seconds
frames_per_buffer_recording_dataset = 1024


# ----------------------------
# Record an audio file. Return the signal in bytes
# ----------------------------
def record_audio(filename):
    
    """
    Record an audio file of given duration.
    :param filename: Name of the output file.
    :param duration: Duration of the recording in seconds.
    :param fs: Sampling rate.
    """
    
    p = pyaudio.PyAudio()

    # Open a new stream for recording
    stream = p.open(format=pyaudio.paInt16,
                    channels=n_channels,
                    rate=fs,
                    input=True,
                    frames_per_buffer=frames_per_buffer_recording_dataset)

    print(f"Recording {filename}...")

    frames = []

    for _ in range(0, int(fs / 1024 * duration_recording_dataset)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recording to a .wav file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    
# ----------------------------
# Load an audio file. Return the signal as a tensor and the sample rate
# ----------------------------

def open_audio(audio_file):
    sig, sr = torchaudio.load(audio_file) # sig = [1, 47104] and sr = 16000 (fs)
    return (sig, sr)

# ----------------------------
# Convert the given audio to the desired number of channels
# ----------------------------

def rechannel(aud, new_channel):
    sig, sr = aud
    
    if (sig.shape[0] == new_channel):
      # Nothing to do
      return aud
    
    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      resig = sig[:1, :]
    else:
      # Convert from mono to stereo by duplicating the first channel
      resig = torch.cat([sig, sig])
    
    return ((resig, sr))


# ----------------------------
# Since Resample applies to a single channel, we resample one channel at a time
# ----------------------------

def resample(aud, newsr):
    sig, sr = aud
    
    if (sr == newsr):
      # Nothing to do
      return aud
    
    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
      retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])
    
    return ((resig, newsr))


# ----------------------------
# Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
# ----------------------------

def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms
    
    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]
    
    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len
    
      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))
    
      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)


# ----------------------------
# Generate a Spectrogram
# ----------------------------

def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
  
    sig,sr = aud
    top_db = 80
    
    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    
    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    
    return (spec) # shape: [1, 64, 94] --> [channels, n_mels, time_steps]




