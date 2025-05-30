import os
import json
import torch
from random import seed
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import PreTrainedTokenizerFast


MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
TOKENIZER_DIR = "model/wordlevel_tokenizer"
DATA_FILE = "data/training_data.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
seed(SEED)
torch.manual_seed(SEED)


tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
vocab_size = tokenizer.vocab_size


def load_dataset(filepath):
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids = tokenizer.encode(f"[BOS] {line} [EOS]")
                if len(ids) <= MAX_LENGTH:
                    examples.append(torch.tensor(ids))
    return examples


dataset = load_dataset(DATA_FILE)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])


def collate(batch):
    padded = pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = padded.clone()
    return padded.to(DEVICE), labels.to(DEVICE)


next_token_id = tokenizer.encode("next")[0]

first_sample = None
rest_samples = []


for sample in train_data:
    if first_sample is None and next_token_id in sample:
        first_sample = sample
    else:
        rest_samples.append(sample)

# New ordered dataset: first the desired sample, then the rest
ordered_train_data = [first_sample] + rest_samples

# Step 2: Use the ordered data without shuffle
train_loader = DataLoader(
    ordered_train_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
)


config = HookedTransformerConfig(
    # Architecture
    n_layers=4,  # 4 transformer layers
    n_heads=8,  # 8 attention heads per layer
    d_model=128,  # embedding/model dimension
    d_head=32,  # head size; d_model = n_heads * d_head
    d_mlp=512,  # MLP dimension (typically 2–4× d_model)
    # Tokenization / vocabulary
    d_vocab=vocab_size,
    # Context
    n_ctx=64,  # context length (sequence length)
    # Activation & Normalization
    act_fn="gelu",  # activation function
    normalization_type="LNPre",  # layer norm before attention/MLP
    # Misc
    attn_only=False,  # include both attention and MLPs
    use_attn_result=True,
    # Positional embeddings
    positional_embedding_type="rotary",
    rotary_base=1000,
)

model = HookedTransformer(config).to(DEVICE)


optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


def compute_accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            for input_seq, label_seq in zip(inputs, labels):
                # Remove padding and special tokens
                input_ids = input_seq.tolist()
                label_ids = label_seq.tolist()

                if tokenizer.pad_token_id in input_ids:
                    pad_idx = input_ids.index(tokenizer.pad_token_id)
                    input_ids = input_ids[:pad_idx]
                    label_ids = label_ids[:pad_idx]

                if tokenizer.eos_token_id in input_ids:
                    eos_idx = input_ids.index(tokenizer.eos_token_id)
                    input_ids = input_ids[:eos_idx]
                    label_ids = label_ids[:eos_idx]

                if tokenizer.bos_token_id in input_ids:
                    input_ids.remove(tokenizer.bos_token_id)
                    label_ids.remove(tokenizer.bos_token_id)

                if len(input_ids) < 2:
                    continue  # Too short

                prompt = [tokenizer.bos_token_id] + input_ids[:-1]
                target = input_ids[-1]

                input_tensor = torch.tensor([prompt], dtype=torch.long).to(DEVICE)
                logits = model(input_tensor)
                pred_id = torch.argmax(logits[0, -1]).item()

                if pred_id == target:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0


print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        logits = model(inputs)
        logits = logits[:, :-1].reshape(-1, vocab_size)
        labels = labels[:, 1:].reshape(-1)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = model(inputs)
            logits = logits[:, :-1].reshape(-1, vocab_size)
            labels = labels[:, 1:].reshape(-1)
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_acc = compute_accuracy(model, val_loader)

    print(
        f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}"
    )
    scheduler.step()


def clean_config_dict(cfg):
    """Convert non-serializable values like torch.dtype to strings."""
    cleaned = {}
    for k, v in cfg.items():
        if isinstance(v, torch.dtype):
            cleaned[k] = str(v)
        elif isinstance(v, Path):
            cleaned[k] = str(v)
        elif isinstance(v, (list, tuple)):
            cleaned[k] = [
                str(item) if isinstance(item, torch.dtype) else item for item in v
            ]
        else:
            cleaned[k] = v
    return cleaned


# Save directory
save_dir = "model/trained_transformerlens_model"
Path(save_dir).mkdir(exist_ok=True)

# Save weights
torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))

# Save config
with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(clean_config_dict(model.cfg.to_dict()), f, indent=2)

print(f"Model saved to {save_dir}")
