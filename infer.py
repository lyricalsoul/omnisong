import torch

from model.device import device
from model.gpt import GPT
from model.grammar_mask import build_allowed_mask
from util import make_path

model_file = make_path("omni.pth")
with open(model_file, "rb") as f:
    ckpt = torch.load(f, map_location=device)

stoi = ckpt['stoi']
itos = ckpt['itos']
vocab_size = len(stoi)

model = GPT(vocab_size=vocab_size)
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()

print("Model loaded from", model_file, "vocabulary size:", vocab_size)


def encode(text):
    return [stoi[token] for token in text.split() if token in stoi]


def decode(indices):
    return ' '.join([itos[idx] for idx in indices])


def sample_logits(logits, temperature=1.0, top_k=None):
    logits = logits / temperature
    if top_k is not None:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(prompt, max_len=256, temperature=1.0, top_p=0.9, debug=False):
    model.eval()

    tokens = []
    for t in prompt.split():
        if t in stoi:
            tokens.append(stoi[t])

    if debug:
        print(f"Starting generation with prompt: '{prompt}'")
        print(f"Encoded tokens: {tokens}")
        print(f"Decoded: {decode(tokens) if tokens else 'Empty'}")

    for step in range(max_len):
        # encode current tokens
        inp_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # get logits from model
        logits = model(inp_ids)[:, -1, :]  # last-token logits
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1).squeeze()

        # apply grammar mask to filter allowed tokens
        allowed_mask = build_allowed_mask(itos, tokens)
        allowed_mask = torch.tensor(
            allowed_mask,
            dtype=torch.float32,
            device=device
        )

        masked = probs * allowed_mask

        if masked.sum().item() == 0:
            break

        masked = masked / masked.sum()

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(masked, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[0] = False

            # create a mask to remove tokens
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            masked[indices_to_remove] = 0
            masked = masked / masked.sum()

        if debug and step < 10:
            allowed_tokens = [itos[i] for i in range(len(allowed_mask)) if allowed_mask[i] > 0]
            print(
                f"Step {step}: {len(allowed_tokens)} allowed tokens, last token: {itos[tokens[-1]] if tokens else 'None'}")

        next_id = torch.multinomial(masked, 1).item()

        tokens.append(next_id)

    return " ".join([itos[t] for t in tokens])
