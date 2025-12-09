import torch
print(torch.__version__)          # e.g., 2.1.0
print(torch.version.cuda)         # CUDA version PyTorch was compiled with
print(torch.backends.cudnn.version())  # cuDNN version
print(torch.cuda.is_available())  # True if GPU is detected
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.current_device())# Current GPU device index