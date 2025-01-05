---
title: "Wake-Word Detection for your AI robot: A Step-by-Step Guide"
publishedAt: "2025-01-08"
lastUpdatedAt: "2025-01-08"
summary: "Learn how to build a wake-word detection system for your devices, exploiting NLP and AI. Understand the key concepts, training process and its application in robotics."
tags: "NLP, AI, Language Processing, Robotics, Machine Learning, Wake-Word Detection"
image: "/images/???????"
readingTime: "20 minutes"
codeRepo: "https://github.com/leo-berte/wake_word_detection"
---

# Wake-Word Detection for your AI robot: A Step-by-Step Guide
In this guide, we will walk you through the process of building a wake word detection system for [Argo](https://argorobot.it/), an AI home robot. Along the way, we will also highlight the general pipeline used in AI projects, making this guide perfect for those new to Artificial Intelligence (AI), Natural Language Processing (NLP), or robotics.

<iframe width="560" height="315" src="https://www.youtube.com/embed/pKa5SEiUZ1g" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


<Image
    src="/images/wake-word/argo_intro.png"
    alt="Argo"
    width={800}
    height={400}
/>
<Caption>
    Argo is an home robot capable of autonomous navigation, entertainment, conversational AI, and more. This type of decide needs to be able to be triggered by voice when asked.
</Caption>

Throughout the development, we’ll focus on the following stages:

1) **Dataset Creation**: Learn how to gather and prepare high-quality data for training your model.
2) **Dataset Augmentation**: Discover techniques to expand and diversify the dataset, enhancing model robustness.
3) **Feature Selection**: Understand how to extract the most relevant features from raw data to boost performance.
4) **Neural Network Architecture**: Learn how to design a model capable of effectively learning patterns from the data.
5) **Training and Evaluation**: Gain insights into optimizing and evaluating the model to ensure it performs at its best.
6) **Deployment**: Understand the steps for preparing the model for integration into real-world applications.

By following this structured approach, you will not only learn how to build a wake word detection system but also gain a solid framework for tackling a wide range of AI challenges. Whether you’re a student, hobbyist, or aspiring professional, this guide will equip you with the skills and knowledge to begin your journey into AI and robotics.

## 1. Problem Overview

**Argo** is a home robot designed for autonomous navigation, conversational AI, entertainment, and much more. As a companion robot, it is essential for Argo to recognize when it is called by its owner, making wake word detection a critical skill. The goal is for Argo to detect the phrase "Hey Argo" reliably, while ignoring background noise or irrelevant speech. Once detected, Argo will quickly switch into attention mode to respond appropriately. 

This is called supervised learning and the idea behind is simple: if you show enough photo to a PC saying: this is a cat or this is a dog, then, when will be shown a new photo, it will be able to recognize if that is  a dog or cat. The same will happen here with audio file containing (or note) the wake word "Hey Argo".

This task relies on a technique called *supervised learning*. The concept is simple: by showing a machine multiple examples of labeled data, such as images of cats and dogs with the corresponding labels, the system learns to recognize patterns. Similarly, by feeding the model audio files containing the wake word "Hey Argo" (or not), it will learn to differentiate between when the wake word is spoken and when it is not.
## 2. Dataset Creation
For our dataset, we recorded two types of audio:

* "positive" samples: 300 recordings of people saying "Hey Argo".

* "negative" samples: 500 recordings of random noise, silence or other words.

Below is a simple script to record 3-second audio samples and save them as .wav files with the corresponding label (positive or negative) embedded in the filename.

```python
import pyaudio
import wave

def record_audio(filename, duration=3, fs=16000):

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True)

    print(f"Recording {filename}...")
    data = stream.read(int(fs * duration))  # Read audio data in a single shot
    print("Finished recording.")

    # stop stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # save audio in .wav
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(fs)
        wf.writeframes(data)

# Example usage
record_audio("positive_sample_01.wav")
```

## 3. Dataset Augmentation
Data augmentation is essential for building a robust wake word detection system. It enhances the model's generalization by simulating real-world conditions such as background noise, pitch variations, and different speaking speeds. By expanding the dataset with augmented samples, the model becomes more resilient to variations in speakers, accents, and environments, ensuring reliable performance in diverse scenarios.

Moreover, collecting and labeling real-world audio data can be time-consuming and expensive. Augmentation reduces the need for extensive data collection by reusing existing samples creatively.

Given a recorded audio, we can apply:

1) Pitch shifting
2) Audio stretching
3) Random noise

```python
import librosa

def pitch_shift(audio_path, output_path, shift):

    """
    Apply pitch shifting to audio file
    """

    y, sr = librosa.load(audio_path, sr=None)  # Load the audio file
    y_shifted = librosa.effects.pitch_shift(y, sr, shift)  # Apply pitch shift
    sf.write(output_path, y_shifted, sr)  # Save the modified audio

def stretch_audio(audio_path, output_path, rate):

    """
    Stretch or compress the audio duration
    """
    
    y, sr = librosa.load(audio_path, sr=None)  # Load the audio file
    y_stretched = librosa.effects.time_stretch(y, rate)  # Apply time stretch
    sf.write(output_path, y_stretched, sr)  # Save the modified audio

def add_noise(audio_path, output_path, noise_factor=0.005):

    """
    Add random noise to the audio file to simulate background interference
    """
    
    y, sr = librosa.load(audio_path, sr=None)  # Load the audio file
    noise_amp = noise_factor * np.random.uniform() * np.amax(y)  # Compute noise amplitude
    y_noisy = y + noise_amp * np.random.normal(size=y.shape)  # Add noise to the audio
    sf.write(output_path, y_noisy, sr)  # Save the noisy audio
```

## 4. Feature Selection: Mel-Spectrogram
Feature selection is a crucial step in building machine learning models, especially when working with audio data. Audio signals are typically very complex and raw waveforms contain a large amount of information, much of which may be irrelevant for a given task. The goal of feature extraction is to transform raw audio into a more compact and informative representation, making it easier for models to learn patterns and make predictions.
In our case, we chose the Mel-Spectrogram as the primary feature representation of the audio signal. This method converts the audio signal into the frequency domain, breaking it down into its spectral components (how much energy exists at various frequencies). 

Basically, given an input audio file, its raw format is typically represented as a one-dimensional array *1 x N* , where:

**N**: The number of time steps or samples in the audio file.

**1**: Each value in the array represents the audio intensity (amplitude) at a specific time step.

<Image
    src="/images/flow-matching/flow_matching_diagram.png"
    alt="Flow Matching Diagram"
    width={800}
    height={400}
/>
<Caption>
    Flow Matching's key components and their interactions: A velocity field defines the flow that generates probability paths. The framework simplifies complex flows with challenging boundary conditions (top) into more manageable conditional flows (middle). Arrows indicate dependencies. The velocity field is learned through loss functions, primarily using Conditional Flow Matching (CFM) loss in practice.
</Caption>

After applying a spectrogram transformation, the audio data is converted into a 2D representation with dimensions *M x K*, where:

**M**: It corresponds to the range of frequencies in the signal (*n_mels*).

**K**: The number of time frames (*total_number_of_samples / hop_length*)

<Image
    src="/images/flow-matching/flow_matching_diagram.png"
    alt="Flow Matching Diagram"
    width={800}
    height={400}
/>
<Caption>
    Flow Matching's key components and their interactions: A velocity field defines the flow that generates probability paths. The framework simplifies complex flows with challenging boundary conditions (top) into more manageable conditional flows (middle). Arrows indicate dependencies. The velocity field is learned through loss functions, primarily using Conditional Flow Matching (CFM) loss in practice.
</Caption>

You can learn more about MEL-spectrogram here: INSERIRE UN LINK.


Snippet to compute the mel-spectrogram:

```
from torchaudio import transforms
import matplotlib.pyplot as plt

def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):

    # Load the audio signal
    sig, sr = aud  # 'sig' is the audio signal, 'sr' is the sampling rate
    
    top_db = 80  # The threshold in decibels for filtering weak signals
    
    # Apply MelSpectrogram transformation
    spec = transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    
    # Convert amplitude to decibels (logarithmic scale)
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    
    return spec  # Shape: [channels, n_mels, time_steps]
```

ELENCO SEGUENTE LO TOGLIEREI, O SPIEGHIAMO BENE COSA è MEL-SPETTROGRAM OPPURE QUESTI DA SOLI HAN POCO SENSO.

* n_mels:
The number of Mel bands (bins) to use in the Mel filterbank. The default value is 64, which works well in most speech recognition tasks. Increasing this value captures more detailed frequency information, but may also introduce noise.
* n_fft:
The size of the window for the Fast Fourier Transform (FFT), which determines the frequency resolution. A larger window gives better frequency resolution but lower time resolution. Typically, values range from 512 to 2048.
* hop_len:
The hop length defines the step size between consecutive frames. This controls how much the frames overlap. If not provided, the function uses n_fft // 2, meaning a 50% overlap. This parameter influences the time resolution and how smooth or detailed the spectrogram will be.
* top_db:
This parameter defines the threshold in decibels. Anything below this value will be suppressed, which helps to focus on the more prominent parts of the spectrogram (ignoring quieter background noise).

## 5. Neural Network Architecture
We analyzed three main architectures: RNN, GRU, and LSTM. Below is a brief theoretical description and the code for each.

### RNN (Recurrent Neural Network)
RNNs calculate temporal representations through a recurrent mechanism. The basic equation is:

$$
h_t = \tanh(W_h x_t + U_h h_{t-1} + b_h)
$$

Where:
- **$h_t$**: Hidden state at time $t$
- **$x_t$**: Input at time $t$
- **$h_{t-1}$**: Hidden state from the previous time step
- **$W_h$, $U_h$, $b_h$**: Weight matrices and bias for the hidden state

Note: The RNN processes the input sequence timestep by timestep, producing a hidden state at each timestep. 
So in the output I would have a matrix  sequence_length, hidden_size
However, here I consider only the last timestep, since it contains all the relevant features extracted for classification

QUA MAGARI DIRE INPUT_SIZE = TOT, OUTPUT = TOT, ...

Snippet:

```python
import torch
import torch.nn as nn

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

# net hyperparameters
batch_size=32
epochs=120
learning_rate=0.00005 
input_size = 64 # n_mels 
hidden_size = 256
output_size = 1  # Output size for binary classification
num_layers = 2

# define the model
model = RNNModel(input_size, hidden_size, output_size, num_layers)


# QUESTA PARTE + SPIEGAZIONE METTIAMOLA SOLO NEL TRAINING/EVAL

# Using BCEWithLogitsLoss instead of CrossEntropyLoss
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

### GRU (Gated Recurrent Unit)
GRUs introduce a gating mechanism to mitigate the vanishing gradient problem. The equations are:

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

$$
\tilde{h}_t = \tanh(W_h x_t + r_t \odot (U_h h_{t-1}) + b_h)
$$

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

Where:

- **$z_t$**: Update gate at time $t$
- **$r_t$**: Reset gate at time $t$
- **$\tilde{h}_t$**: Candidate hidden state at time $t$
- **$h_t$**: Final hidden state at time $t$
- **$W_z, W_r, W_h$**: Input weight matrices
- **$U_z, U_r, U_h$**: Hidden state weight matrices
- **$b_z, b_r, b_h$**: Bias terms

Snippet:

```python
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
input_size = 64 # n_mels
hidden_size = 256*2
output_size = 1  # Output size for binary classification
num_layers = 1

# define the model
model = GRUModel(input_size, hidden_size, output_size, num_layers)
```

### LSTM (Long Short-Term Memory)
LSTMs handle long-term dependencies with a memory cell. The equations are:

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$

$$
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

Where:

- **$f_t$**: Forget gate at time $t$
- **$i_t$**: Input gate at time $t$
- **$o_t$**: Output gate at time $t$
- **$\tilde{c}_t$**: Candidate cell state at time $t$
- **$c_t$**: Cell state at time $t$
- **$h_t$**: Hidden state at time $t$
- **$W_f, W_i, W_c, W_o$**: Input weight matrices
- **$U_f, U_i, U_c, U_o$**: Hidden state weight matrices
- **$b_f, b_i, b_c, b_o$**: Bias terms

Snippet:

```python
class LSTMModel(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
            
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        
        return out

# net hyperparameters
batch_size = 32
epochs = 50
learning_rate = 0.0001
input_size = 64 # n_mels 
hidden_size = 256 * 2
output_size = 1  # Output size for binary classification
num_layers = 1

# define model
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
```

## 6. Training and Evaluation

Now we'll finally use the dataset to train our model. We can split the dataset in a part used for training (80%), and the remaining part (20%) for evaluating our model.

```python
    # Random split of 80:20 between training and validation
    num_items = len(dataset)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(dataset, [num_train, num_val])
    print("train set: ", num_train)
    print("valid set: ", num_val)
```

The training uses cross-entropy loss (what is?) and the Adam optimizer. Moreover, at the end of each epoch, we compute the average loss. This is a key parameter to monitor during training, since it shall cecrease over time meaning the error performed on the trainins set is decreasing.

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(train_dl):
    
    # For plotting
    train_loss_values = []
    valid_loss_values = []

    # Training loop
    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        
        for i, (inputs, labels) in enumerate(train_dl):

            # inference
            outputs = model(inputs)
            
            # compute loss
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # check loss after each epoch
        average_loss = epoch_loss / len(train_dl)
        train_loss_values.append(average_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')

    # Plot loss
    plt.plot(range(1, epochs + 1), train_loss_values, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
```

Once the training is completed, we can test the trained weights on the validation set to measure accuracy and other metrics. 
Basically, we use our neural net to forecast whether there is the activation word or not in the input audio from the validation set. Since we have labels, we can detect whether the neural net was right or not, and compute its accuracy.
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def eval_model(val_dl):
    
    model.eval()
    valid_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        
        for inputs, labels in val_dl:
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            
            # Calculate predictions
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs >= 0.5).float()
            
            # Store labels and predictions for metric calculation
            all_labels.extend(labels.numpy()) # converting to NumPy
            all_predictions.extend(predicted_labels.numpy()) # converting to NumPy
    
    valid_average_loss = valid_loss / len(val_dl)

    # Plot loss
    plt.plot(range(0, len(train_loss_values)), train_loss_values, marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
```

## 7. Deployment

The final model is optimized using dynamic quantization:

```python
# total record time in seconds
record_time=60 

# audio stream parameters
fs=16000 # sample rate
chunk_duration = 1 # each read window
feed_duration = 2.944   # the total feed length (the ones which will generate the input for the NN)
chunk_samples = int(fs * chunk_duration) 
feed_samples = int(fs * feed_duration)


# callback for the audio stream data   
def callback(in_data, frame_count, time_info, status): 

    global run, data 

    # transform bytes in numpy arrays
    new_data = np.frombuffer(in_data, dtype='int16')
    # deque the oldest chunk_samples
    data = np.roll(data, -chunk_samples) 
    # enque the newest chunk_samples
    data[feed_samples-chunk_samples:] = new_data[:] 
    
    # save data in the queue
    que.put(data)
    
    return (in_data, pyaudio.paContinue)


if __name__ == '__main__':
    
    print('Start recording...')
    
    # define and start stream
    que = Queue() 
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
                      
            # get the input features     
            sgram = spectro_gram((data,fs))
            
            # eval the input
            is_wake_word_detected = eval_model(sgram)
            
            # display result
            if (is_wake_word_detected == True):
                print("hey argo detected")
            else:
                print("random word")
            
    except (KeyboardInterrupt, SystemExit):
        
        print("Exiting... Bye.")
        stream.stop_stream()
        stream.close()
        run = False
    
    # final clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
```
Now, the model is ready to be executed in real-time on your laptop... but wouln't be better to see it running on a real Argo? 

## 8. Key Works and Citations

- **Chen et al. (2018)**: *Neural Ordinary Differential Equations.*
- **Grathwohl et al. (2018)**: *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models.*
- **Lipman et al. (2022)**: *Flow Matching for Generative Modeling.*
