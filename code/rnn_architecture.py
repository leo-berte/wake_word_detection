# ----------------------------------------
# Code defining different NN architectures
# ----------------------------------------

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


# define different NN architectures
NN_MODEL_TYPE = "gru" # "rnn" - "gru" - "lstm"


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
    epochs=2
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







# class GRUModel(nn.Module):
#     def __init__(self, input_size, Tx, hidden_size, output_size):
#         super(GRUModel, self).__init__()
        
#         # Convolutional layer
#         self.conv1d = nn.Conv1d(in_channels=Tx, out_channels=196, kernel_size=15, stride=4)
#         self.batch_norm1 = nn.BatchNorm1d(196)
#         self.dropout1 = nn.Dropout(0.8)
        
#         # First GRU layer
#         self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
#         self.dropout2 = nn.Dropout(0.8)
#         self.batch_norm2 = nn.BatchNorm1d(Tx)
        
#         # Second GRU layer
#         self.gru2 = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
#         self.dropout3 = nn.Dropout(0.8)
#         self.batch_norm3 = nn.BatchNorm1d(Tx)
        
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # Convolutional layer
#         x = self.conv1d(x)
#         x = self.batch_norm1(x)
#         x = F.relu(x)
#         x = self.dropout1(x)
        
#         # First GRU layer
#         x, _ = self.gru1(x.permute(0, 2, 1))
#         x = self.dropout2(x)
#         x = self.batch_norm2(x)
        
#         # Second GRU layer
#         x, _ = self.gru2(x)
#         x = self.dropout3(x)
#         x = self.batch_norm3(x)
        
#         out = self.fc(x[:, -1, :])

#         return out

    
    
# # net hyperparameters

# batch_size=32
# epochs=70
# learning_rate=0.0001   

# # example_spectrogram, _ = next(iter(train_dl))
# input_size = 64 # example_spectrogram.shape[2]  # n_mels --> defined in "dataset_processing.py
# hidden_size = 256*2
# output_size = 1  # Output size for binary classification
# Tx = 93
# model = GRUModel(input_size, Tx, hidden_size, output_size)

# # Using BCEWithLogitsLoss instead of CrossEntropyLoss
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)