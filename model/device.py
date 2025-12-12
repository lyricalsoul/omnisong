import torch

if torch.cuda.is_available():
    # for PCs with nvidia gpus (CUDA cores)
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # for macs specially with Apple Silicon: metal performance shaders
    device = torch.device("mps")
else:
    # fallback to cpu
    device = torch.device("cpu")
    print("GPU not available. Inference will use CPU which will be slower.")
