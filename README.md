# wake_word_detection

Wake-word detection module for Argo, an AI Smart Assistant Robot (argorobot.it). 

Collected the dataset with positive words "Hey Argo" and negative words (random words or background noise). 
Performed audio post-processing, NN architecture definition, training, hyper-parameters tuning and final deploy on live audio stream.

===============================

# code: 

- build_dataset.py: code used to create the dataset and augment (eventually) the dataset.
- audio_processing_lib.py: this code contains the functions related to audio processing (record, compute spectrogram, ..).
- dataset_processing_lib.py: this code contains the functions related to dataset processing (create and augment dataset, get dataloader, ..).
- nn_architecture.py: code defining different NN architectures.
- wake_word_training.py: code for training.
- wake_word_eval.py: code for running inference over a singli input audio file.
- wake_word_live.py: code for recording audio live and print the wake word when it is detected.

# dataset: 

- originals: positive words "Hey Argo" and negative words (random words or background noise).
- augemented: takes the file in "originals" and augment them with pitch shifting and audio stretching.

# models: 

weights of the NN trained on different datasets (augmented vs orginals) and different architectures (RNN, GRU, LSTM)

===============================

# Run

- wake_word_live.py: keep the audio stream for N seconds and in the meanwhile it prints "hey Argo detected" or "random word" anytime it hears someone speaking.
- build_dataset.py: create a dataset by recording both positive words ("hey Argo") and negative words (random words or background noise).
- 
===============================

Argo website: argorobot.it
