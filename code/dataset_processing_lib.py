# ------------------------------------------------------------------------------------------------
# This code contains the functions related to dataset processing (create and augment dataset, get dataloader, ..)
# ------------------------------------------------------------------------------------------------


import os
import random
import soundfile as sf
import numpy as np
import librosa
from torch.utils.data import DataLoader, Dataset, random_split


# custom libraries
from audio_processing_lib import *
from rnn_architecture import *


def collect_dataset(flag_wake_word, samples_per_class, wake_word, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    # Get existing files to avoid overwriting
    existing_files = os.listdir(output_dir)
    wake_word_files = [f for f in existing_files if f.startswith(wake_word)]
    not_wake_word_files = [f for f in existing_files if f.startswith('not_' + wake_word)]

    # Determine the next index to use
    wake_word_count = len(wake_word_files)
    not_wake_word_count = len(not_wake_word_files)
    print("wake_word_count: ", wake_word_count)
    print("not_wake_word_count: ", not_wake_word_count)

    for i in range(samples_per_class):
        if flag_wake_word == "positive":
            # Determine file index ensuring unique names
            wake_word_filename = os.path.join(output_dir, f'{wake_word}_{wake_word_count + i}.wav')
            # Record the audio
            record_audio(wake_word_filename)
        elif flag_wake_word == "negative":
            # Determine file index ensuring unique names
            not_wake_word_filename = os.path.join(output_dir, f'not_{wake_word}_{not_wake_word_count + i}.wav')
            # Record the audio
            record_audio(not_wake_word_filename)

    print("Dataset collection complete.")


def augment_dataset(input_dir, output_dir):
    """
    Perform data augmentation on audio files in the input directory.
    Augmentations include pitch shifting, time stretching, and adding noise.
    Augmented files are saved in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            filepath = os.path.join(input_dir, filename)
            y, sr = librosa.load(filepath, sr=None)

            # Pitch shifting
            pitch_shift = random.choice([-1, 1])  # Shift pitch by up to 1 semitone
            y_pitch_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_shift)
            pitch_filename = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_pitch.wav')
            sf.write(pitch_filename, y_pitch_shifted, sr)

            # Time stretching
            time_stretch = random.uniform(1.0, 1.3)  # Make the audio shorter (faster voice)
            y_stretched = librosa.effects.time_stretch(y=y, rate=time_stretch)
            padding = len(y) - len(y_stretched)
            pad_left = padding // 2
            pad_right = padding - pad_left
            y_resampled = np.pad(y_stretched, (pad_left, pad_right), 'constant') # Resample the stretched audio to the original number of samples
            stretch_filename = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_stretch.wav')
            sf.write(stretch_filename, y_resampled, sr)

            # # Adding noise
            # noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            # y_noisy = y + noise_amp * np.random.normal(size=y.shape)
            # noise_filename = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_noise.wav')
            # sf.write(noise_filename, y_noisy, sr)

    print("Data augmentation complete.")


# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    
  def __init__(self, file_paths, labels):
    self.labels = labels
    self.file_paths = file_paths
    self.duration = 3000 # desired max milliseconds for the audio
    self.sr = 16000 # desired frames per second (fs)
    self.channel = 1 # desired channels

            
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.file_paths)    
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):
      
      file_path = self.file_paths[idx]
      label = self.labels[idx]

      aud = open_audio(file_path) 
      # reaud = AudioUtil.resample(aud, self.sr)
      # rechan = AudioUtil.rechannel(reaud, self.channel) 
      # dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
      sgram = spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)

      return sgram, label



def get_datasets(dataset_path):
    
    # Load dataset
    file_paths = []  # it will contain the path to every item in dataset
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.wav'):
            file_paths.append(os.path.join(dataset_path, filename))
            label = 0 if 'not_hey_argo' in filename else 1
            labels.append(label)
    
    dataset = SoundDS(file_paths, labels)

    # Random split of 80:20 between training and validation
    num_items = len(dataset)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(dataset, [num_train, num_val])
    print("train set: ", num_train)
    print("valid set: ", num_val)
    
    # Create training and validation data loaders
    dataset_dl = DataLoader(dataset, batch_size, shuffle=True)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    
    return dataset_dl, train_dl, val_dl