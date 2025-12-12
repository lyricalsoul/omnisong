import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import MusicDataset
from model.device import device
from model.gpt import GPT

TRAIN_FILE = "training_plain.txt"
EPOCHS = 50
GRADIENT_CLIP = 1.0
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4


def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


with open(TRAIN_FILE) as f:
    text = f.read()

# 90/10 train/val split
lines = text.strip().split('\n')
split_idx = int(len(lines) * 0.9)
train_text = '\n'.join(lines[:split_idx])
val_text = '\n'.join(lines[split_idx:])

train_dataset = MusicDataset(train_text)
val_dataset = MusicDataset(val_text)

val_dataset.stoi = train_dataset.stoi
val_dataset.itos = train_dataset.itos
val_dataset.data = [train_dataset.stoi.get(token, 0) for token in val_text.split()]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

vocab_size = len(train_dataset.stoi)

# gradient checkpointing to save memory
model = GPT(vocab_size=vocab_size, gradient_checkpointing=True).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                              weight_decay=0.01)  # decay should be changed probs because loss is kinda high
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

best_loss = float('inf')

if __name__ == "__main__":
    start_time = time.time()

    avg_train_loss = 0.0
    avg_val_loss = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        total_grad_norm = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", ncols=100)

        optimizer.zero_grad()
        grad_norm = 0.0

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            # we scale loss down by ACCUMULATION_STEPS and then we unscale when logging
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            total_loss += loss.item() * ACCUMULATION_STEPS

            # update weights every ACCUMULATION_STEPS
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                grad_norm = compute_grad_norm(model)
                total_grad_norm += grad_norm
                # we clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}',
                'grad': f'{grad_norm:.4f}' if (batch_idx + 1) % ACCUMULATION_STEPS == 0 else 'acc'
            })

        avg_train_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc="Validation", ncols=100, leave=False)
        with torch.no_grad():
            for x, y in val_pbar:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s.")
        print(
            f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # if validation loss improved, save model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'stoi': train_dataset.stoi,
                'itos': train_dataset.itos
            }, "omni.pth")
            print(f"New best model saved (val_loss: {best_loss:.4f})")

        scheduler.step()  # finally update LR

    # save final model
    torch.save({
        'epoch': EPOCHS - 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'stoi': train_dataset.stoi,
        'itos': train_dataset.itos
    }, "omni_s.pth")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes.")
