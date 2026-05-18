"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import torch.nn.functional as F
from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.true_dist = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        assert logits.size(1) == self.vocab_size
        true_dist = logits.clone().detach()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        mask = torch.nonzero(target == self.pad_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(F.log_softmax(logits, dim=1), true_dist)


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_tokens = 0
    
    for i, batch in enumerate(data_iter):
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_y = tgt[:, 1:]
        
        src_mask = make_src_mask(src, pad_idx=1).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=1).to(device)
        
        if is_train:
            optimizer.zero_grad()
            
        out = model(src, tgt_input, src_mask, tgt_mask)
        
        loss = loss_fn(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1))
        
        if is_train:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                
        total_loss += loss.item()
        total_tokens += (tgt_y != 1).sum().item()
        
    return total_loss / total_tokens if total_tokens > 0 else 0


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src).to(device)
    
    for i in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = F.softmax(out[:, -1], dim=-1)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).type_as(src).to(device)], dim=1)
        if next_word == end_symbol:
            break
            
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    model.eval()
    from torchtext.data.metrics import bleu_score
    
    targets = []
    outputs = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            
            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_i, pad_idx=1).to(device)
                
                try:
                    sos_idx = tgt_vocab['<sos>']
                    eos_idx = tgt_vocab['<eos>']
                except:
                    sos_idx = 2
                    eos_idx = 3
                
                out = greedy_decode(model, src_i, src_mask, max_len, sos_idx, eos_idx, device)
                
                out_tokens = []
                for idx in out.squeeze(0):
                    if idx.item() == eos_idx:
                        break
                    if idx.item() not in [sos_idx, 1]:  # skip pad and sos
                        try:
                            itos = getattr(tgt_vocab, 'get_itos', lambda: tgt_vocab.itos)()
                            out_tokens.append(itos[idx.item()])
                        except:
                            out_tokens.append(str(idx.item()))
                
                tgt_tokens = []
                for idx in tgt[i]:
                    if idx.item() == eos_idx:
                        break
                    if idx.item() not in [sos_idx, 1]:
                        try:
                            itos = getattr(tgt_vocab, 'get_itos', lambda: tgt_vocab.itos)()
                            tgt_tokens.append(itos[idx.item()])
                        except:
                            tgt_tokens.append(str(idx.item()))
                            
                outputs.append(out_tokens)
                targets.append([tgt_tokens])
                
    return bleu_score(outputs, targets) * 100


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': {
            'src_vocab_size': model.src_vocab_size,
            'tgt_vocab_size': model.tgt_vocab_size,
            'd_model': model.d_model,
            'N': len(model.encoder.layers),
            'num_heads': model.encoder.layers[0].self_attn.num_heads,
            'd_ff': model.encoder.layers[0].feed_forward.linear1.out_features,
            'dropout': model.pos_encoder.dropout.p
        }
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    import os
    if not os.path.exists(path):
        return 0
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0)


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    import wandb
    import dataset
    from torch.nn.utils.rnn import pad_sequence
    from lr_scheduler import NoamScheduler
    import torch.nn.functional as F
    
    config = {
        'd_model': 256,
        'N': 3,
        'num_heads': 8,
        'd_ff': 512,
        'dropout': 0.1,
        'batch_size': 64,
        'num_epochs': 30,
        'warmup_steps': 4000,
        'smoothing': 0.1
    }
    
    # disable W&B logging for local testing, can be re-enabled
    wandb.init(project="da6401-a3", config=config, mode='disabled')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ds = dataset.Multi30kDataset()
    try:
        vocab_de, vocab_en = dataset.load_vocab(gdrive_id="1I5H6o_sRs6xlu-hIDuovFdxLYfYYpGpH")
        ds.vocab_de = vocab_de
        ds.vocab_en = vocab_en
    except Exception as e:
        print(f"Could not load vocab due to error: {e}. Building from dataset instead...")
        vocab_de, vocab_en = ds.build_vocab()
    
    data = ds.process_data()
    train_data = data[:int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)):int(0.9 * len(data))]
    test_data = data[int(0.9 * len(data)):]
    
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)
        src_batch = pad_sequence(src_batch, padding_value=vocab_de['<pad>'], batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=vocab_en['<pad>'], batch_first=True)
        return src_batch, tgt_batch
        
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    print(len(vocab_en))
    print(len(vocab_de))
    model = Transformer(
        src_vocab_size=len(vocab_de),
        tgt_vocab_size=len(vocab_en),
        d_model=config['d_model'],
        N=config['N'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        checkpoint_path=None
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, config['d_model'], config['warmup_steps'])
    loss_fn = LabelSmoothingLoss(len(vocab_en), vocab_en['<pad>'], config['smoothing']).to(device)
    
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving best checkpoint.")
            save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt")
        
        # Save last checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint_last.pt")
        
    bleu = evaluate_bleu(model, test_loader, vocab_en, device)
    wandb.log({'test_bleu': bleu})
    print(f"Test BLEU: {bleu:.2f}")


if __name__ == "__main__":
    run_training_experiment()