import os
import json
import torch
import wandb
import argparse
import importlib.util
from random import seed
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from transformers import PreTrainedTokenizerFast
from transformer_lens import HookedTransformer, HookedTransformerConfig

from utils.train_utils import (
    clean_config_dict,
    logit_diff_score,
    load_dataset,
    compute_accuracy,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
seed(SEED)
torch.manual_seed(SEED)


def main(mps):
    tokens_seen = 0  # Global token counter
    tokenizer = PreTrainedTokenizerFast.from_pretrained(mps.tokenizer_dir)
    vocab_size = tokenizer.vocab_size

    dataset = load_dataset(mps.data_path, tokenizer)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    next_token_id = tokenizer.encode("next")[0]

    next_train_data = [sample for sample in train_data if next_token_id in sample]
    last_train_data = [sample for sample in train_data if next_token_id not in sample]

    def collate(batch):
        padded = pad_sequence(
            batch, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = padded.clone()
        return padded.to(DEVICE), labels.to(DEVICE)

    next_train_loader = DataLoader(
        next_train_data, batch_size=mps.batch_size, shuffle=True, collate_fn=collate
    )
    last_train_loader = DataLoader(
        last_train_data, batch_size=mps.batch_size, shuffle=True, collate_fn=collate
    )

    val_loader = DataLoader(
        val_data, batch_size=mps.batch_size, shuffle=False, collate_fn=collate
    )

    config = HookedTransformerConfig(
        n_layers=mps.n_layers,
        n_heads=mps.n_heads,
        d_model=mps.d_model,
        d_head=mps.d_head,
        d_mlp=mps.d_mlp if hasattr(mps, "d_mlp") else None,
        d_vocab=vocab_size,
        n_ctx=mps.n_ctx,
        act_fn=mps.act_fn,
        normalization_type=mps.normalization_type,
        attn_only=mps.attn_only,
        use_attn_result=mps.use_attn_result,
        positional_embedding_type=mps.positional_embedding_type,
        rotary_base=mps.rotary_base if hasattr(mps, "rotary_base") else None,
    )

    model = HookedTransformer(config).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=mps.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    wandb.init(
        project=mps.project_name,
        name=mps.run_name,
        config={
            "epochs": mps.epochs,
            "batch_size": mps.batch_size,
            "lr": mps.learning_rate,
            "tokenizer": mps.tokenizer_dir,
            **clean_config_dict(model.cfg.to_dict()),
        },
    )

    print("Starting training...")
    for epoch in range(mps.epochs):
        total_loss_A = total_loss_B = 0
        total_logit_diff_A = total_logit_diff_B = 0

        model.train()
        for phase, train_loader in [
            ("A_last_token", last_train_loader),
            ("B_next_token", next_train_loader),
        ]:
            phase_loss = 0
            phase_logit_diff = 0

            for inputs, labels in train_loader:
                logits = model(inputs)
                logits_trimmed = logits[:, :-1]
                labels_trimmed = labels[:, 1:]

                flat_logits = logits_trimmed.reshape(-1, vocab_size)
                flat_labels = labels_trimmed.reshape(-1)

                loss = loss_fn(flat_logits, flat_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_token_count = (
                    (labels_trimmed != tokenizer.pad_token_id).sum().item()
                )
                tokens_seen += batch_token_count
                logit_diff = logit_diff_score(logits_trimmed, labels_trimmed)

                phase_loss += loss.item()
                phase_logit_diff += logit_diff

                if phase.startswith("A"):
                    total_loss_A = phase_loss
                    total_logit_diff_A = phase_logit_diff
                else:
                    total_loss_B = phase_loss
                    total_logit_diff_B = phase_logit_diff

                wandb.log(
                    {
                        "tokens_seen": tokens_seen,
                        f"train/loss_{phase}": loss.item(),
                        f"train/logit_diff_{phase}": logit_diff,
                        "epoch": epoch + 1,
                        "train/phase": phase,
                    }
                )

            avg_phase_loss = phase_loss / len(train_loader)
            avg_phase_logit_diff = phase_logit_diff / len(train_loader)
            print(
                f"Epoch {epoch+1} | Phase {phase}: Loss = {avg_phase_loss:.4f}, LogitDiff = {avg_phase_logit_diff:.4f}"
            )

        # --- Validation ---
        model.eval()
        val_loss, val_logit_diff = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                logits = model(inputs)
                logits_trimmed = logits[:, :-1]
                labels_trimmed = labels[:, 1:]

                flat_logits = logits_trimmed.reshape(-1, vocab_size)
                flat_labels = labels_trimmed.reshape(-1)

                loss = loss_fn(flat_logits, flat_labels)
                val_loss += loss.item()
                val_logit_diff += logit_diff_score(logits_trimmed, labels_trimmed)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_logit_diff = val_logit_diff / len(val_loader)
        val_acc = compute_accuracy(model, val_loader, tokenizer, DEVICE)

        wandb.log(
            {
                "epoch": epoch + 1,
                "val/loss": avg_val_loss,
                "val/accuracy": val_acc,
                "val/logit_diff": avg_val_logit_diff,
            }
        )

        avg_train_loss = (total_loss_A + total_loss_B) / (
            len(last_train_loader) + len(next_train_loader)
        )
        avg_train_logit_diff = (total_logit_diff_A + total_logit_diff_B) / (
            len(last_train_loader) + len(next_train_loader)
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/avg_loss": avg_train_loss,
                "train/avg_logit_diff": avg_train_logit_diff,
            }
        )
        scheduler.step()

    # Save directory
    Path(mps.save_dir).mkdir(exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), os.path.join(mps.save_dir, "model_weights.pth"))

    # Save config
    with open(os.path.join(mps.save_dir, "config.json"), "w") as f:
        json.dump(clean_config_dict(model.cfg.to_dict()), f, indent=2)

    print(f"Model saved to {mps.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/full_model_rope.py",
        help="Path to the model parameters configuration file.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

    spec = importlib.util.spec_from_file_location(
        config_path.replace(".py", "").replace("/", ".").replace("..", ""), config_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    main(module)
