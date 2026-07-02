"""
smoke_test.py  (linux-port helper — NOT part of the original S³ release)

Cheapest possible sanity check: build the ISRUC model (CBraMod backbone, random
init) + the S³ SNN segmenter and print parameter counts — WITHOUT any dataset or
pretrained weights. Mirrors the param-count print in main.py:230-264.

Purpose:
  1. Verify the schoffelen/gwilliams import bug is fixed (imports resolve).
  2. Verify model + SNN construct on this machine (CUDA/spikingjelly OK).
  3. Confirm the SNN (SAS) is tiny (~9K params, matches paper's +0.008M for S³)
     and the CBraMod backbone is ~4-5M (paper reports ~4.000M).

Run (from repo root, inside the py3.10 env):
    python smoke_test.py
Expected: 'SMOKE OK' at the end, SNN params ~9K, backbone params ~4-5M.
"""
import argparse
import torch
import numpy as np

from models.snn import SAS
from models import model_isruc


def build_isruc_args():
    """Replicate main.py's ISRUC config block (no argparse/data needed)."""
    args = argparse.Namespace()
    # model / task
    args.model = 'cbramod'
    args.load_lbm = False            # random-init backbone: no cbramod-base.pth needed
    args.foundation_dir = None
    # ISRUC hardcoded config (main.py:73-83)
    args.n_classes = 5
    args.n_subjects = 100
    args.n_channels = 6
    args.sr = 200
    args.fps = 1
    # Brain2Event / SNN knobs (main.py defaults)
    args.C = 0.2
    args.n_slice = 1
    return args


def main():
    args = build_isruc_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[smoke] device={device}, torch={torch.__version__}, cuda_avail={torch.cuda.is_available()}")

    eeg_model = model_isruc.Model(args)
    snn_model = SAS(args)

    # same param split as main.py:230-236 / 262-264
    backbone_params, other_params = [], []
    for name, param in eeg_model.named_parameters():
        (backbone_params if "backbone" in name else other_params).append(param)

    n_ann = sum(p.numel() for p in other_params)
    n_snn = sum(p.numel() for p in snn_model.parameters())
    n_backbone = sum(p.numel() for p in backbone_params)

    print(f"[smoke] The ann (head+seq+classifier) contains {n_ann:,} parameters "
          f"({n_ann/1e6:.3f} M).")
    print(f"[smoke] The snn (S3 segmenter) contains   {n_snn:,} parameters "
          f"({n_snn/1e6:.4f} M).")
    print(f"[smoke] The backbone (CBraMod) contains   {n_backbone:,} parameters "
          f"({n_backbone/1e6:.3f} M).")

    # sanity expectations (loose): SNN tiny, backbone few-M
    ok = True
    if not (5_000 <= n_snn <= 20_000):
        print(f"[smoke][WARN] SNN param count {n_snn} outside expected ~9K band.")
        ok = False
    if not (3_000_000 <= n_backbone <= 6_000_000):
        print(f"[smoke][WARN] backbone param count {n_backbone} outside expected ~4-5M band.")
        ok = False

    # tiny forward through the SNN segmenter on a fake ISRUC-shaped batch (no data files)
    try:
        B, C, T = 2, args.n_channels, args.sr * 30   # 30s window @200Hz
        fake = torch.randn(B, C, T, device=device)
        snn_model = snn_model.to(device)
        with torch.no_grad():
            spike_idxes = snn_model(fake)
        print(f"[smoke] SNN forward OK on fake [B={B},C={C},T={T}] -> "
              f"{len(spike_idxes)} spike-index lists, e.g. len(batch0)={len(spike_idxes[0])}")
    except Exception as e:
        print(f"[smoke][WARN] SNN forward failed: {type(e).__name__}: {e}")
        ok = False

    print("SMOKE OK" if ok else "SMOKE DONE (with warnings — see above)")


if __name__ == '__main__':
    main()
