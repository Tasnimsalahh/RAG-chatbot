import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1e6} MB")
