# ----------------------------------------
# Code defining different NN architectures
# ----------------------------------------

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.quantization as quantization
import torch



# TODO: fix QAT



# quantization settings
QUANTIZATION_AWARE_TRAINING_FLAG = False # True - False

# Quantization: convert the weights and activations in int8 instead of float32,
# so that the inference is faster on embedded devices. 

def prepare_for_qat(model):
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    model = quantization.prepare_qat(model, inplace=True)
    return model


def convert_to_quantized(model):
    model.eval()
    model = quantization.convert(model, inplace=True)
    return model


# global device flag
DEVICE_FLAG = "cuda" if torch.cuda.is_available() else "cpu"  # automatically choose 'cuda' if available, otherwise 'cpu'

# define different NN architectures
NN_MODEL_TYPE = "transformer" # "rnn" - "gru" - "gru_advanced" - "lstm" - "transformer"
                 

if NN_MODEL_TYPE == "rnn":
    
    class RNNModel(nn.Module):
        
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(RNNModel, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) 
            self.fc = nn.Linear(hidden_size, output_size)
        
        
        def forward(self, x):
            # input: batch_size, sequence_length, features --> output: batch_size, sequence_length, hidden_size
            out, _ = self.rnn(x) 
            # input: batch_size, hidden_size --> output: batch_size, output_size
            out = self.fc(out[:, -1, :]) 
            return out
    
    # Note: The RNN processes the input sequence timestep by timestep, producing a hidden state at each timestep. 
    # So in the output I would have a matrix  sequence_length, hidden_size
    # However, here I consider only the last timestep, since it contains all the relevant features extracted for classification
    
    # net hyperparameters
    batch_size=32
    epochs=120
    learning_rate=0.00005 
    
    # example_spectrogram, _ = next(iter(train_dl))
    input_size = 64 # example_spectrogram.shape[2]  # n_mels --> defined in "dataset_processing.py
    hidden_size = 256
    output_size = 1  # Output size for binary classification
    num_layers = 2
    model = RNNModel(input_size, hidden_size, output_size, num_layers)
    model = model.to(DEVICE_FLAG) # move model to the selected device (CPU or CUDA)
    
    # Using BCEWithLogitsLoss instead of CrossEntropyLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



elif NN_MODEL_TYPE == "gru":

    class GRUModel(nn.Module):
        
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(GRUModel, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            
            # add quantization steps
            if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
                self.quant = quantization.QuantStub()
                self.dequant = quantization.DeQuantStub()
        
        
        def forward(self, x):
            
            if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
                x = self.quant(x)
                
            out, _ = self.gru(x)
            out = self.fc(out[:, -1, :])
            
            if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
                out = self.dequant(out)
            
            return out
        
        
    # net hyperparameters
    batch_size=32
    epochs=100
    learning_rate=0.0001   
    
    # example_spectrogram, _ = next(iter(train_dl))
    input_size = 64 # example_spectrogram.shape[2]  # n_mels --> defined in "dataset_processing.py
    hidden_size = 256*2
    output_size = 1  # Output size for binary classification
    num_layers = 1
    model = GRUModel(input_size, hidden_size, output_size, num_layers)
    
    if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
        model = prepare_for_qat(model)
        
    # move model to the selected device (CPU or CUDA)
    model = model.to(DEVICE_FLAG)
        
    # Using BCEWithLogitsLoss instead of CrossEntropyLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)






elif NN_MODEL_TYPE == "gru_advanced":  
      
    class GRUAdvancedModel(nn.Module):
        
        def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
            super(GRUAdvancedModel, self).__init__()
            
            # First GRU layer
            self.gru1 = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
            self.dropout1 = nn.Dropout(dropout_prob)  
            self.batch_norm1 = nn.BatchNorm1d(hidden_size)  
            
            # Second GRU layer
            self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
            self.dropout2 = nn.Dropout(dropout_prob)  
            self.batch_norm2 = nn.BatchNorm1d(hidden_size)  
    
            # Fnal dropout
            self.dropout_final = nn.Dropout(dropout_prob)
            
            # Fully connected layer 
            self.fc = nn.Linear(hidden_size, output_size)
            
            # add quantization steps
            if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
                self.quant = quantization.QuantStub()
                self.dequant = quantization.DeQuantStub()
            
            
        def forward(self, x):
            
            if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
                x = self.quant(x)
                
            # First GRU
            out, _ = self.gru1(x)
            # dropout e batch norm
            out = self.dropout1(out)
            out = self.batch_norm1(out.transpose(1, 2)).transpose(1, 2) # batch_norm needs this format: [batch_size, hidden_size, time_steps]
            
            # Second GRU
            out, _ = self.gru2(out)
            # dropout e batch norm
            out = self.dropout2(out)
            out = self.batch_norm2(out.transpose(1, 2)).transpose(1, 2) # batch_norm needs this format: [batch_size, hidden_size, time_steps]
            
            # final dropout
            out = self.dropout_final(out)
            
            # fully connected layer
            out = self.fc(out[:, -1, :])  # FC applied oly to the final time step
            
            if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
                out = self.dequant(out)
            
            return out 
    
    

    # net hyperparameters
    batch_size=32
    epochs=2
    learning_rate=0.0001   
    dropout_prob=0.8
    
    # example_spectrogram, _ = next(iter(train_dl))
    input_size = 64 # example_spectrogram.shape[2]  # n_mels --> defined in "dataset_processing.py
    hidden_size = 256*2
    output_size = 1  # Output size for binary classification
    num_layers = 1
    model = GRUAdvancedModel(input_size, hidden_size, output_size, num_layers, dropout_prob)
    
    if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
        model = prepare_for_qat(model)
    
    # move model to the selected device (CPU or CUDA)
    model = model.to(DEVICE_FLAG)
    
    # Using BCEWithLogitsLoss instead of CrossEntropyLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    
    
elif NN_MODEL_TYPE == "lstm":

    class LSTMModel(nn.Module):
        
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(LSTMModel, self).__init__()
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            
            # add quantization steps
            if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
                self.quant = quantization.QuantStub()
                self.dequant = quantization.DeQuantStub()
        
        
        def forward(self, x):
            
            if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
                x = self.quant(x)
                
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            
            if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
                out = self.dequant(out)
            
            return out
        
        
    # net hyperparameters
    batch_size = 32
    epochs = 50
    learning_rate = 0.0001
    
    # example_spectrogram, _ = next(iter(train_dl))
    input_size = 64 # example_spectrogram.shape[2]  # n_mels --> defined in "dataset_processing.py
    hidden_size = 256 * 2
    output_size = 1  # Output size for binary classification
    num_layers = 1
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    
    if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
        model = prepare_for_qat(model)
    
    # move model to the selected device (CPU or CUDA)
    model = model.to(DEVICE_FLAG)
    
    # loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)



elif NN_MODEL_TYPE == "transformer":
    
    # NOTE: for wakeword I can just use encoder, since I do not have to generate anything as output with the decoder
    
    class TransformerEncoderModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, nhead, output_size):
            super(TransformerEncoderModel, self).__init__()
    
            self.embedding = nn.Linear(input_size, hidden_size)  # Layer di embedding
            self.encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
            self.fc = nn.Linear(hidden_size, output_size)  # Layer di output
    
    
        def forward(self, x):
            # x dovrebbe avere forma [batch_size, seq_length, input_size]
            x = self.embedding(x)  # Passa attraverso il layer di embedding
            x = x.permute(1, 0, 2)  # Cambia la forma in [seq_length, batch_size, hidden_size] per il Transformer
            x = self.transformer_encoder(x)  # Passa attraverso il blocco encoder
            x = x.mean(dim=0)  # Media delle sequenze (opzionale, puoi anche usare solo l'ultimo output)
            x = self.fc(x)  # Passa attraverso il layer di output
            return x
    
    
    # net hyperparameters
    batch_size = 32
    epochs = 50
    learning_rate = 0.0001
    
    # example_spectrogram, _ = next(iter(train_dl))
    input_size = 64 # example_spectrogram.shape[2]  # n_mels --> defined in "dataset_processing.py
    hidden_size = 256
    output_size = 1  # Output size for binary classification
    num_layers = 2 # number of stacked encoders
    nhead = 4  # number of heads
    
    # define the model
    model = TransformerEncoderModel(input_size, hidden_size, num_layers, nhead, output_size).to(DEVICE_FLAG)
    
    # move model to the selected device (CPU or CUDA)
    model = model.to(DEVICE_FLAG)
    
    # loss
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    

    
