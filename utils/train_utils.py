import torch
from pathlib import Path


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


def logit_diff_score(logits, labels):
    # logits: [batch, seq, vocab], labels: [batch, seq]
    # logit of true label - top incorrect logit
    probs = torch.nn.functional.log_softmax(logits, dim=-1)
    label_logits = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    top_logits, top_indices = probs.topk(2, dim=-1)
    top_incorrect = torch.where(
        top_indices[..., 0] == labels, top_logits[..., 1], top_logits[..., 0]
    )
    logit_diff = (label_logits - top_incorrect).mean().item()
    return logit_diff


def load_dataset(filepath, tokenizer):
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids = tokenizer.encode(f"[BOS] {line} [EOS]")
            examples.append(torch.tensor(ids))
    return examples


def compute_accuracy(model, data_loader, tokenizer, device):
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

                input_tensor = torch.tensor([prompt], dtype=torch.long).to(device)
                logits = model(input_tensor)
                pred_id = torch.argmax(logits[0, -1]).item()

                if pred_id == target:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0
