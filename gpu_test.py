import torch
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.tensor([1, 2, 3], device=device)
print(tensor)

