# ---------------------------------------------------------------------------------------------
# Code for dynamic quantization of the model: convert the weights in int8 instead of float32,
# so that the inference is faster on embedded devices. Here the activation functions remain float32.
# The alternative is QAT (quantization aware training), where the training takes into account
# quantization directly.
# ---------------------------------------------------------------------------------------------


import torch
from torch import nn

# pre-trained model
model = torch.load('model.pt')

# apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,  # model
    {nn.LSTM, nn.Linear},  # layers
    dtype=torch.qint8  # type of quantization (int8)
)

# save the final model with quantization
torch.save(quantized_model, 'quantized_model.pt')
