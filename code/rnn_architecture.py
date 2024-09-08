# ----------------------------------------
# Code defining different NN architectures
# ----------------------------------------

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


# define different NN architectures
NN_MODEL_TYPE = "gru_advanced" # "rnn" - "gru" - "gru_advanced" - lstm"


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
    
    # Using BCEWithLogitsLoss instead of CrossEntropyLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



elif NN_MODEL_TYPE == "gru":

    class GRUModel(nn.Module):
        
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(GRUModel, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            out, _ = self.gru(x)
            out = self.fc(out[:, -1, :])
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
            
            
        def forward(self, x):
            
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
            
            return out
    

    # net hyperparameters
    batch_size=32
    epochs=100
    learning_rate=0.0001   
    dropout_prob=0.8
    
    # example_spectrogram, _ = next(iter(train_dl))
    input_size = 64 # example_spectrogram.shape[2]  # n_mels --> defined in "dataset_processing.py
    hidden_size = 256*2
    output_size = 1  # Output size for binary classification
    num_layers = 1
    model = GRUAdvancedModel(input_size, hidden_size, output_size, num_layers, dropout_prob)
    
    # Using BCEWithLogitsLoss instead of CrossEntropyLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    
    
elif NN_MODEL_TYPE == "lstm":
    
    pass