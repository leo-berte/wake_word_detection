# ----------------------------------------
# Code for loading dataset and training
# ----------------------------------------
# 
# Explain spectrogram: https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505
# RNN model: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
# GRU model: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
# GRU CODE: https://github.com/JamesMcGuigan/coursera-deeplearning-specialization/blob/master/05_Sequence_Models/Week%203/Trigger%20word%20detection/Trigger_word_detection_v1a.ipynb
# LSTM model: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
#
# ----------------------------------------


# NOTES:
# Come faccio training e tuning? ciclo for in cui vario n_layers, batch_size, lr e cos'altro?
# per ogni ciclo, che metriche salvo per fare poi confronti? validation loss, accuracy, grafici, ...?



import matplotlib.pyplot as plt

# custom libraries
from dataset_processing_lib import *
from nn_architecture import *


def train_model(train_dl, val_dl):
    
    # For plotting
    train_loss_values = []
    valid_loss_values = []

    # Training loop
    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        
        for i, (inputs, labels) in enumerate(train_dl):
            
            # labels = labels.float().unsqueeze(1) # equivalent to view(-1,1)
            labels = labels.float().view(-1, 1).to(DEVICE_FLAG)  # batch_size, 1            
            
            # Ensure spectrograms have the correct shape for NN
            inputs = inputs.squeeze(1).permute(0, 2, 1).to(DEVICE_FLAG)  # from [batch_size, 1, n_mels, time_steps] to [batch_size, time_steps, n_mels]
            
            # inference
            outputs = model(inputs)
            
            # compute loss
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        average_loss = epoch_loss / len(train_dl)
        train_loss_values.append(average_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')
        
        # Validation phase (optional but recommended)
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, labels in val_dl:
                labels = labels.float().view(-1, 1).to(DEVICE_FLAG)
                inputs = inputs.squeeze(1).permute(0, 2, 1).to(DEVICE_FLAG)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        
        valid_average_loss = valid_loss /  len(val_dl)
        valid_loss_values.append(valid_average_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')
        
    # Plot loss
    plt.plot(range(1, epochs + 1), train_loss_values, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Plot loss
    plt.plot(range(1, epochs + 1), valid_loss_values, marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def eval_model(val_dl):
    
    model.eval()
    valid_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in val_dl:
            labels = labels.float().view(-1, 1).to(DEVICE_FLAG)
            inputs = inputs.squeeze(1).permute(0, 2, 1).to(DEVICE_FLAG)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            
            # Calculate predictions
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs >= 0.5).float()
            print("labels: ", predicted_labels.shape)
            print("len: ", len(val_dl))
            
            # Store labels and predictions for metric calculation
            all_labels.extend(labels.cpu().numpy()) # move back to CPU before converting to NumPy
            all_predictions.extend(predicted_labels.cpu().numpy()) # move back to CPU before converting to NumPy
    
    valid_average_loss = valid_loss / len(val_dl)

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f'Validation Loss: {valid_average_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

    return valid_average_loss, accuracy, precision, recall, f1




#######################################################################################

if __name__ == '__main__':
    
    # use the original dataset or the augmented dataset
    dataset_choice = 'originals' # 'augmented' 'originals'
    dataset_path = os.path.join('../dataset', dataset_choice)
    
    # get the training set and validation set
    dataset_dl, train_dl, val_dl = get_datasets(dataset_path) 
    
    # # check batch dimensions
    # for i, batch in enumerate(dataset_dl):
    #     data, labels = batch
    #     print(f"Batch {i+1}: Dati shape {data.shape}, Labels shape {labels.shape}")
        
    # train the model
    train_model(train_dl, val_dl)    
    
    # eval the model on validation set
    valid_average_loss, accuracy, precision, recall, f1 = eval_model(val_dl)
    
    # Convert the model to quantized version
    if (QUANTIZATION_AWARE_TRAINING_FLAG == True):
        model = convert_to_quantized(model)
    
    # it is good to re-evaluate the quantization model on the validation set
    if QUANTIZATION_AWARE_TRAINING_FLAG == True:
        valid_average_loss_quant, accuracy_quant, precision_quant, recall_quant, f1_quant = eval_model(val_dl)
    
    # Save the model
    model_data = f'{NN_MODEL_TYPE}_{dataset_choice}_{valid_average_loss:.2f}_{accuracy:.2f}_{batch_size}_{epochs}_{hidden_size}_{num_layers}_{learning_rate}.pth' 
    model_data_name = os.path.join('../models', model_data)
    torch.save(model.state_dict(), model_data_name)
    
    
   