import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU detected!")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Total GPUs:", torch.cuda.device_count())
else:
    print("No GPU detected. Using CPU instead.")
