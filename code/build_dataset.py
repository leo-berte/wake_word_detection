# ----------------------------------------------------------------------------
# This code is used to create the dataset and augment (eventually) the dataset
# ----------------------------------------------------------------------------


# custom libraries
from audio_processing_lib import *
from dataset_processing_lib import *


# NOTES:
    
# 1) Usa hey argo con almeno 5 voci diverse
# 2) Come faccio data augementation? dato audio orginale --> originale + noise, originale + pitch, originale + stretch
# 3) Per i negative examples registra frase anche come Hey Alexa, Hey Bro, .. Astro, Ramarro, Chiasso, ....
# 4) Possono esserci anche 1000 dati presi da internet per negative words, e solo 100 di positive words prese da noi
#    Altre idee senò per aumentare dati? Da internet si trova qualche dataset? Devono essere però di 3 secondi gli audio
# 5) Registrati mentre fai una cena e la gente parla



if __name__ == '__main__':
    
    # flags
    flag_augment_dataset = True # True False
    flag_collect_dataset = False # True False
    flag_wake_word = "negative"  # negative: collect negative samples - positive: collect positive samples
    
    # paramas
    wake_word='hey_argo'
    collect_output_dir='../dataset/originals'
    augment_input_dir='../dataset/originals'
    augment_output_dir='../dataset/augmented'
    samples_per_class = 30

    # Collect the original dataset
    if flag_collect_dataset == True:
        collect_dataset(flag_wake_word, samples_per_class, wake_word, collect_output_dir)

    # Augment the dataset
    if flag_augment_dataset == True:
        augment_dataset(augment_input_dir, augment_output_dir)