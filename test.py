import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Current CUDA Device:", torch.cuda.current_device())
print("CUDA Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
