"""
Training script for Volume-Set MTPP model on BFNX data.
Supports both Mac (CPU/MPS) and CUDA devices.
"""

import os
import sys
import json
import argparse
import time
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Make tensorboard optional
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: Tensorboard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from volume_set_mtpp.models.volume_set_mtpp import VolumeSetMTPP, create_volume_set_mtpp
from volume_set_mtpp.training.data_loader import create_bfnx_dataloaders


def get_device(device_type: str = 'auto') -> torch.device:
    """
    Get the appropriate device for training.

    Args:
        device_type: 'auto', 'cuda', 'mps', or 'cpu'

    Returns:
        torch.device object
    """
    if device_type == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
            print("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_type)
        print(f"Using specified device: {device_type}")

    return device


def create_model(num_channels: int, config: dict, device: torch.device) -> VolumeSetMTPP:
    """
    Create Volume-Set MTPP model.

    Args:
        num_channels: Number of event types
        config: Model configuration
        device: Device to place model on

    Returns:
        VolumeSetMTPP instance
    """
    # Use the factory function from volume_set_mtpp module
    model = create_volume_set_mtpp(
        num_channels=num_channels,
        config=config,
        device=device,
        use_volume=config.get('use_volume', True),
        intensity_type=config.get('intensity_type', 'dynamic')
    )

    return model


def compute_loss(model, batch, device):
    """
    Compute negative log-likelihood loss for a batch.

    Args:
        model: VolumeSetMTPP instance
        batch: Dictionary containing batch data
        device: Device to run on

    Returns:
        loss tensor
    """
    # Use the model's built-in compute_loss method
    loss, metrics = model.compute_loss(batch, device)
    return loss


def train_epoch(model, train_loader, optimizer, device, epoch, writer=None, loss_writer=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # STRICT: any batch failure or non-finite value aborts the run with a
        # nonzero exit -- a silently skipped batch or a truncated epoch must
        # never masquerade as a successful training run.
        loss = compute_loss(model, batch, device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch} batch {batch_idx}: {loss.item()}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite grad norm at epoch {epoch} batch {batch_idx}: {grad_norm}")
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

        global_step = epoch * len(train_loader) + batch_idx
        if writer:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
        if loss_writer:
            loss_writer.writerow({'split': 'train', 'epoch': epoch, 'batch': batch_idx, 'global_step': global_step, 'loss': float(loss.item())})

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_epoch_tbptt(model, train_loader, optimizer, device, epoch, writer=None, loss_writer=None):
    """One epoch of TBPTT training: batches arrive in stream order (lane
    batching, StatefulBFNXLoader) and the decoder state is CARRIED across
    batches -- detached, so gradients truncate at window boundaries while the
    state values flow from the start of each lane's stream. Lanes flagged in
    reset_mask (lane start / zone boundary) restart from the decoder's learned
    init_state instead of a mid-stream carried state."""
    model.train()
    dec = model.decoder
    L, H = dec.num_layers, dec.recurrent_hidden_size
    init = dec.init_state.detach()                        # [L, H] learned cold-start state
    carried = None                                        # [B, L, H]
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train TBPTT]", total=len(train_loader))

    for batch_idx, batch in enumerate(pbar):
        reset = batch.pop("reset_mask").to(device)
        b = int(reset.shape[0])
        if carried is None:
            carried = init.unsqueeze(0).expand(b, -1, -1).clone().to(device)
        old = carried.clone()
        old[reset] = init.to(device)
        loss, _ = model.compute_loss(batch, device, old_states=old)

        # STRICT: non-finite values abort with nonzero exit (see train_epoch).
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch} batch {batch_idx}: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite grad norm at epoch {epoch} batch {batch_idx}: {grad_norm}")
        optimizer.step()

        # hand-off: packed [B, D*] -> layer states [B, L, H] (held anchors are
        # recomputed by the next window's event pass; see ARCHITECTURE.md)
        packed = model._last_final_state
        carried = packed[:, :L * H].reshape(b, L, H)

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': loss.item()})
        global_step = epoch * len(train_loader) + batch_idx
        if writer:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
        if loss_writer:
            loss_writer.writerow({'split': 'train', 'epoch': epoch, 'batch': batch_idx, 'global_step': global_step, 'loss': float(loss.item())})

    return total_loss / max(n_batches, 1)


def evaluate(model, val_loader, device, epoch, writer=None, loss_writer=None):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for batch_idx, batch in enumerate(pbar):
            # STRICT: validation failures abort too -- a val loss averaged over
            # a silently reduced batch set corrupts model selection.
            loss = compute_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite val loss at epoch {epoch} batch {batch_idx}")
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            if loss_writer:
                loss_writer.writerow({'split': 'val_batch', 'epoch': epoch, 'batch': batch_idx, 'global_step': epoch * len(val_loader) + batch_idx, 'loss': float(loss.item())})

    avg_loss = total_loss / len(val_loader)

    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train Volume-Set MTPP on BFNX data')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/events/bfnx',
                        help='Directory containing BFNX event files')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to load (for testing)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for tensorized BFNX cache; default is DATA_DIR/.tensor_cache')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Force rebuild of tensorized BFNX cache from JSONL')

    # Model arguments
    parser.add_argument('--channel-emb-size', type=int, default=64,
                        help='Channel embedding size')
    parser.add_argument('--time-emb-size', type=int, default=128,
                        help='Time embedding size (must equal recurrent hidden size for HawkesDecoder)')
    parser.add_argument('--recurrent-hidden', type=int, default=128,
                        help='Recurrent hidden size')
    parser.add_argument('--dominating-rate', type=float, default=100.0,
                        help='Dominating rate for thinning algorithm')
    parser.add_argument('--time-loss-weight', type=float, default=1.0,
                        help='Weight for ground time negative log-likelihood')
    parser.add_argument('--set-loss-weight', type=float, default=1.0,
                        help='Weight for Bernoulli event-set negative log-likelihood')
    parser.add_argument('--lob-state-input', action='store_true',
                        help='condition heads on continuous LOB book features from the data')
    parser.add_argument('--volume-head', action='store_true',
                        help='enable the explicit per-channel log-volume prediction head')
    parser.add_argument('--no-volume-input-scaling', action='store_true',
                        help='Disable the legacy volume-intensity input scaling '
                             '(sets config use_volume=False instead of the hardcoded True)')
    parser.add_argument('--volume-loss-weight', type=float, default=1.0,
                        help='weight of the log-volume NLL term when --volume-head is set')
    parser.add_argument('--volume-head-detach', action='store_true',
                        help='stop-gradient between the volume head and the recurrent state: '
                             'volumes are predicted but their loss cannot reshape the dynamics')
    parser.add_argument('--subcritical-weight', type=float, default=0.0,
                        help='>0 enables the Hawkes-subcriticality penalty on the s2p2 decoder')
    parser.add_argument('--subcritical-rho-max', type=float, default=0.0,
                        help='threshold on the decoder branching proxy for the subcriticality penalty')
    parser.add_argument('--intensity-link', choices=['softplus', 'sigmoid'], default='softplus',
                        help='sigmoid = bounded intensity lambda_max*sigma(z): smooth rate '
                             'saturation with a global subcriticality bound')
    parser.add_argument('--lambda-max', type=float, default=0.0,
                        help='intensity ceiling (events/s) for --intensity-link sigmoid')
    parser.add_argument('--mark-head', choices=['bernoulli', 'categorical'], default='bernoulli',
                        help='categorical = single-mark softmax for event-driven data (drops the set condition)')
    parser.add_argument('--mc-compensator', action='store_true',
                        help='unbiased Monte-Carlo compensator (uniform samples over the window, '
                             'Mei & Eisner style) instead of the biased endpoint rule lambda(t_i^-)*dt_i')
    parser.add_argument('--mc-samples', type=int, default=32,
                        help='MC sample count per sequence for --mc-compensator')
    parser.add_argument('--potential-head', action='store_true',
                        help='2-D (activity, imbalance) potential-flow feedback head: '
                             'local-supercritical/global-stable bursts, momentum mean-reversion, asymmetry')
    parser.add_argument('--trust-region-cap', action='store_true',
                        help='radial cap on the readout state: identity inside the trained '
                             'envelope, clipped outside (closed-loop saturation)')
    parser.add_argument('--trust-region-k', type=float, default=1.0,
                        help='cap radius multiplier on the tracked envelope')
    parser.add_argument('--subcritical-closed', action='store_true',
                        help='exact closed-form branching ratio on the s2p2 query path '
                             '(top-layer kick x readout / decay); gauge-free, no quadrature')
    parser.add_argument('--subcritical-empirical', action='store_true',
                        help='measure the branching ratio on the intensity function (impulse response) '
                             'instead of weight norms; immune to reparameterization gaming')
    parser.add_argument('--subcritical-horizon', type=float, default=20.0,
                        help='integration horizon (s) for the empirical branching ratio')
    parser.add_argument('--subcritical-nseq', type=int, default=4,
                        help='batch subsample size for the empirical branching ratio')
    parser.add_argument('--subcritical-detach', action='store_true',
                        help='representation-safe penalty: trunk states detached, hinge tunes only the intensity head')
    parser.add_argument('--threes-weight', type=float, default=0.0,
                        help='weight of the 3S/PIT level-calibration term (compensator moments -> Exp(1))')
    parser.add_argument('--set-loss-reduction', choices=['sum', 'mean-labels'], default='sum',
                        help='sum = paper Bernoulli likelihood; mean-labels = average BCE over labels for balancing')
    parser.add_argument('--tbptt', action='store_true',
                        help='Stateful (TBPTT) training: batches walk the stream in order and the '
                             'decoder state is carried across windows (detached at boundaries). '
                             'Forces stride = seq-length; S2P2-family decoders only.')
    parser.add_argument('--ptp-dim', type=int, default=8,
                        help='Per-type latent dim d for the per-type s2p2 baseline (pct-lstm)')
    parser.add_argument('--target-rate', type=float, default=1.8, dest='target_rate',
                        help='Target mean event rate (events/s); initializes the SS2P2 rate-head '
                             'scale. -1 = measure it from the TRAIN split only (no val/test '
                             'leakage into the initialization)')
    parser.add_argument('--decoder-type',
                        choices=['hawkes', 'rmtpp', 's2p2', 'ss2p2', 'lstm', 'sahp', 'ct-lstm', 'pct-lstm', 'ptp-s2p2'],
                        default='hawkes',
                        help='Decoder/backbone: SS2P2 (ours), or baselines: S2P2 diagonal SSM, '
                             'Neural Hawkes CT-LSTM (hawkes/ct-lstm), RMTPP LSTM, plain LSTM, '
                             'SAHP causal attention, per-type parallel CT-LSTM (pct-lstm)')
    parser.add_argument('--sahp-heads', type=int, default=4,
                        help='Number of attention heads for --decoder-type sahp')
    parser.add_argument('--sahp-layers', type=int, default=2,
                        help='Number of transformer encoder layers for --decoder-type sahp')
    parser.add_argument('--s2p2-readout', choices=['state', 'output'], default='state',
                        help="output = paper-faithful: heads read the LayerNorm'd stack output "
                             "(rate-bounded); queries evolve all layers. state = legacy raw top state.")
    parser.add_argument('--s2p2-layers', type=int, default=2,
                        help='Number of stacked latent linear Hawkes/SSM layers for --decoder-type s2p2')
    parser.add_argument('--s2p2-scan', action='store_true', default=True,
                        help='parallel prefix-scan state computation for the '
                             'S2P2/SS2P2 decoders (O(log N) depth instead of a '
                             'sequential per-event loop; numerically equivalent). '
                             'ON by default; --no-s2p2-scan restores the loop.')
    parser.add_argument('--no-s2p2-scan', dest='s2p2_scan', action='store_false',
                        help='sequential per-event loop (legacy; ~184x slower on GPU)')
    parser.add_argument('--s2p2-dropout', type=float, default=0.0,
                        help='Dropout in S2P2 residual blocks')
    parser.add_argument('--no-s2p2-input-dependent-dynamics', action='store_true',
                        help='Disable input-dependent S2P2 decay gates')
    parser.add_argument('--ss2p2-wnorm-cap', type=float, default=6.0,
                        help='SS2P2 G1 rate: l1 cap on the readout weight ||w||_1; '
                             'sets the two-sided bound lambda in (softplus(b-cap), softplus(b+cap))')
    parser.add_argument('--ss2p2-mark-hidden', type=int, default=0,
                        help='SS2P2 rate-neutral mark MLP hidden size (0 = recurrent hidden)')

    # Training arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='Sequence length')
    parser.add_argument('--stride', type=int, default=10,
                        help='Stride for sliding window')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader workers; default is 2 on GPU/MPS, 0 on CPU')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip final test-set evaluation for faster tuning runs')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Disable best/checkpoint .pt writes for speed sweeps')
    parser.add_argument('--allow-tf32', action='store_true',
                        help='Enable TF32 matmul/cuDNN on CUDA GPUs for faster A100/H100 training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible train/val/test split order and model init')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='experiments/bfnx',
                        help='Output directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log-dir', type=str, default='logs/bfnx',
                        help='Tensorboard log directory')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f'Using random seed: {args.seed}')

    # Setup device
    device = get_device(args.device)
    if args.allow_tf32 and device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print('Enabled CUDA TF32 matmul/cuDNN fast path')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Save configuration
    config = {
        'channel_embedding_size': args.channel_emb_size,
        'time_embedding_size': args.time_emb_size,
        'recurrent_hidden_size': args.recurrent_hidden,
        'dominating_rate': args.dominating_rate,
        'dyn_dom_buffer': 4,
        'use_volume': (not args.no_volume_input_scaling),
        'intensity_type': 'dynamic',
        'time_loss_weight': args.time_loss_weight,
        'set_loss_weight': args.set_loss_weight,
        'set_loss_reduction': args.set_loss_reduction,
        'decoder_type': args.decoder_type,
        'ptp_dim': args.ptp_dim,
        'target_rate': args.target_rate,
        'tbptt': args.tbptt,
        's2p2_readout': args.s2p2_readout,
        's2p2_layers': args.s2p2_layers,
        's2p2_dropout': args.s2p2_dropout,
        's2p2_scan': bool(args.s2p2_scan),
        's2p2_input_dependent_dynamics': (not args.no_s2p2_input_dependent_dynamics),
        'ss2p2_wnorm_cap': args.ss2p2_wnorm_cap,
        'ss2p2_mark_hidden': (args.ss2p2_mark_hidden or None),
        'sahp_heads': args.sahp_heads,
        'sahp_layers': args.sahp_layers,
        'volume_head': args.volume_head,
        'volume_loss_weight': args.volume_loss_weight,
        'volume_head_detach': args.volume_head_detach,
        'subcritical_weight': args.subcritical_weight,
        'subcritical_rho_max': args.subcritical_rho_max,
        'intensity_link': args.intensity_link,
        'lambda_max': args.lambda_max,
        'mark_head': args.mark_head,
        'mc_compensator': args.mc_compensator,
        'mc_samples': args.mc_samples,
        'potential_head': args.potential_head,
        'trust_region_cap': args.trust_region_cap,
        'trust_region_k': args.trust_region_k,
        'subcritical_closed': args.subcritical_closed,
        'subcritical_empirical': args.subcritical_empirical,
        'subcritical_horizon': args.subcritical_horizon,
        'subcritical_nseq': args.subcritical_nseq,
        'subcritical_detach': args.subcritical_detach,
        'threes_weight': args.threes_weight,
        'lob_state_input': args.lob_state_input,
        'lob_state_dim': 6,
        'seed': args.seed
    }

    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {config_path}")

    # Create dataloaders
    print("\nLoading BFNX data...")
    if args.tbptt and args.stride != args.seq_length:
        print(f"TBPTT: forcing stride {args.stride} -> {args.seq_length} (non-overlapping windows; "
              "end-state of window t = start-state of window t+1)")
        args.stride = args.seq_length
        config['stride'] = args.seq_length
    train_loader, val_loader, test_loader, event_mapping = create_bfnx_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.seq_length,
        stride=args.stride,
        max_files=args.max_files,
        num_workers=(args.num_workers if args.num_workers is not None else (2 if device.type != 'cpu' else 0)),
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        stateful_train=args.tbptt
    )

    print(f"\nDataset Statistics:")
    print(f"  Number of event types: {event_mapping.num_events}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Train-split-only target rate (no leakage into the rate-head init)
    if args.target_rate is not None and args.target_rate < 0:
        ds = getattr(train_loader, 'dataset', None)
        if ds is None or not hasattr(ds, 'split_rate'):
            raise SystemExit('--target-rate -1 needs the train dataset to expose split_rate()')
        args.target_rate = float(ds.split_rate())
        config['target_rate'] = args.target_rate
        with open(config_path, 'w') as f:      # re-save: config.json was written pre-loaders
            json.dump(config, f, indent=2)
        print(f"Target rate measured from TRAIN split: {args.target_rate:.4f} ev/s")

    # Create model
    print("\nCreating model...")
    model = create_model(event_mapping.num_events, config, device)
    if args.tbptt and not (hasattr(model.decoder, 'init_state')
                           and hasattr(model.decoder, '_initial_layer_states')):
        raise SystemExit(f"--tbptt requires an S2P2-family decoder (state hand-off via "
                         f"old_states); decoder_type={args.decoder_type!r} is unsupported")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Create tensorboard writer if available
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(args.log_dir)
        print(f"Tensorboard logging to: {args.log_dir}")
    else:
        print("Tensorboard not available - continuing without logging")

    # CSV loss history. TensorBoard is optional on the cluster, so keep a lightweight artifact that can always be plotted.
    loss_history_path = os.path.join(args.output_dir, 'loss_history.csv')
    loss_history_file = open(loss_history_path, 'w', newline='')
    loss_writer = csv.DictWriter(loss_history_file, fieldnames=['split', 'epoch', 'batch', 'global_step', 'loss'])
    loss_writer.writeheader()

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 50}")

        # Train
        train_fn = train_epoch_tbptt if args.tbptt else train_epoch
        train_loss = train_fn(model, train_loader, optimizer, device, epoch, writer, loss_writer)
        loss_writer.writerow({'split': 'train_epoch', 'epoch': epoch, 'batch': -1, 'global_step': epoch, 'loss': float(train_loss)})
        loss_history_file.flush()
        print(f"Average train loss: {train_loss:.4f}")

        # Validate
        val_loss = evaluate(model, val_loader, device, epoch, writer, loss_writer)
        loss_writer.writerow({'split': 'val_epoch', 'epoch': epoch, 'batch': -1, 'global_step': epoch, 'loss': float(val_loss)})
        loss_history_file.flush()
        print(f"Average validation loss: {val_loss:.4f}")

        # Update scheduler
        scheduler.step(val_loss)

        # Save best model unless this is a speed-only sweep
        if (not args.no_checkpoint) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'event_mapping': event_mapping
            }, best_path)
            print(f"Saved best model with val loss: {val_loss:.4f}")
        elif args.no_checkpoint and val_loss < best_val_loss:
            best_val_loss = val_loss

        # Save checkpoint
        if (not args.no_checkpoint) and epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'event_mapping': event_mapping
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch}")

    # Test evaluation
    test_loss = None
    if args.skip_test:
        print("\nSkipping final test evaluation (--skip-test).")
    else:
        print("\n" + "=" * 50)
        print("Final Test Evaluation")
        print("=" * 50)

        if args.no_checkpoint:
            print('Skipping final test evaluation because --no-checkpoint leaves no best_model.pt to reload.')
        else:
            # Load best model
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            test_loss = evaluate(model, test_loader, device, epoch=0, loss_writer=loss_writer)
            loss_history_file.flush()
            print(f"Test loss: {test_loss:.4f}")

    # Save test results
    results = {
        'test_loss': test_loss,
        'best_val_loss': best_val_loss,
        'final_epoch': args.epochs,
        'config': config,
        'skip_test': args.skip_test
    }

    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete! Results saved to {args.output_dir}")
    if writer is not None:
        writer.close()
    loss_history_file.close()


if __name__ == "__main__":
    main()