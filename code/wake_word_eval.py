# ----------------------------------------------
# Code for running inference over an audio file
# ----------------------------------------------


import os
import torchaudio
import torch


# custom libraries
from audio_processing_lib import *
from nn_architecture import *

    
# load the model
model.load_state_dict(torch.load('../models/wake_word_gru_0.04_0.95_32_50_512_1_0.0001.pth'))
model.eval()
    
    
def eval_model(sgram):
    
    with torch.no_grad():

        input_sgram = sgram.squeeze(1).permute(0, 2, 1)
        # print("input_audio_nn_shape in eval code: ", input_sgram.shape)
        output = model(input_sgram)
        
        # Calculate predictions
        predicted_prob = torch.sigmoid(output)
        predicted_label = (predicted_prob >= 0.5).float()
        if predicted_label == 1.0:
            return True
        else:
            return False

            


# this code shall be a library, but I can run inference on a single audio setting the falg = True
TEST_EVAL_FLAG = False # True - False 
    

if (TEST_EVAL_FLAG == True):
    
    # give a name to the test audio file
    test_audio_filename = os.path.join('../dataset', 'test_audio.wav') 
    
    # Record the audio
    record_audio(test_audio_filename)
    
    # get the input features
    sig, sr = torchaudio.load(test_audio_filename) # sig = [1, 47104] and sr = 16000 (fs)
    print(sig.dtype)
    print(sig.shape)
    
    # compute spectrogram
    sgram = spectro_gram((sig, sr))
    
    # eval the input
    is_wake_word_detected = eval_model(sgram)
    
    # display result
    if (is_wake_word_detected == True):
        print("hey argo detected")
    else:
        print("random word")