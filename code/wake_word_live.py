# -------------------------------------------------------------------------------
# Code for recording audio live and print the wake word when it is detected
# -------------------------------------------------------------------------------

import pyaudio
import numpy as np
from random import randint
import time
import os
import wave
from queue import Queue

# custom libraries
from audio_processing_lib import *
from dataset_processing_lib import *
from nn_architecture import *
from wake_word_eval import *


# variables
silence_threshold= 30 # magnitude of sound of silence
record_time=60 # total record time in seconds

# audio stream parameters
chunk_duration = 1 # each read window
feed_duration = 2.944   # the total feed length (the ones which will generate the input for the NN)
chunk_samples = int(fs * chunk_duration) 
feed_samples = int(fs * feed_duration)

# flags
SAVE_AUDIO_FLAG = False # True - False : save the audio to test it in a second moment 
DEBUG_AUDIO_FLAG = False # True - False : save the audio to test it in a second moment

# Initialize a counter for saving the files
file_counter = 0



# callback for the audio stream data   
def callback(in_data, frame_count, time_info, status): # bytes: signal # int: number of samples of the signal

    global run, timeout, data 
    
    if time.time() > timeout:
        run = False

    # transform bytes in numpy arrays
    new_data = np.frombuffer(in_data, dtype='int16')
    
    # skip the loop if there is no noise in the environment
    if np.abs(new_data).mean() < silence_threshold:
        print('0')
        return (in_data, pyaudio.paContinue)
    else:
        print('1')
        
        data = np.roll(data, -chunk_samples) # deque the oldest chunk_samples
        data[feed_samples-chunk_samples:] = new_data[:] # enque the newest chunk_samples
        
        # save data in the queue
        que.put(data)
        
        return (in_data, pyaudio.paContinue)




###############################################################################

if __name__ == '__main__':
    
    print('Start recording...')
    
    # define and start stream
    que = Queue() 
    timeout = time.time() + record_time # half a minute
    data = np.zeros(feed_samples, dtype='int16') # data buffer for input
    run = True
    
    # open pyaudio
    p = pyaudio.PyAudio()
    
    # set up and start stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=n_channels,
                    rate=fs, # samples per second
                    input=True,
                    frames_per_buffer=chunk_samples, # samples contained in the chunk window
                    stream_callback=callback)
    
    stream.start_stream()
    
    
    try:
        
        while run:
            
            # get current audio segment to input in the NN
            data = que.get()
            
            # Convert data in pytorch sensor and normalize to align with torchaudio.load() function used during training
            data_tensor = torch.tensor(data, dtype=torch.float32) / 32768.0
            
            # Aggiungi una dimensione per il canale
            data_tensor = data_tensor.unsqueeze(0)  # Forma diventa [n_channels, num_samples]
            
            # print("data_tensor shape: ", data_tensor.shape)
            
            # get the input features     
            sgram = spectro_gram((data_tensor,fs))
            
            # print("sgram type: ", sgram.shape)
            
            # eval the input
            is_wake_word_detected = eval_model(sgram)
            
            # display result
            if (is_wake_word_detected == True):
                print("hey argo detected")
            else:
                print("random word")
            
            if SAVE_AUDIO_FLAG == True:
                # Save the raw audio data into a WAV file
                wav_filename = f"chunk_{file_counter}.wav"
                with wave.open(wav_filename, 'wb') as wf:
                    wf.setnchannels(1)  # Mono audio
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(fs)  # Sample rate
                    wf.writeframes(data.tobytes())  # Write data to file
        
                print(f"Saved {wav_filename}")
                file_counter += 1
            
            if DEBUG_AUDIO_FLAG == True:
                print("TEST AUDIO START")
                sig_test, sr_test = torchaudio.load(wav_filename)
                sgram_test = spectro_gram((sig_test,sr_test))
                is_wake_word_detected = eval_model(sgram_test)
                if (is_wake_word_detected == True):
                    print("hey argo detected")
                else:
                    print("random word")
                print("TEST AUDIO END")
            
    except (KeyboardInterrupt, SystemExit):
        
        print("Exiting... Bye.")
        stream.stop_stream()
        stream.close()
        timeout = time.time()
        run = False
    
    # final clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    
    
    
