"""
Tối ưu Trojan trigger trên Mac (MPS/CPU) từ surrogate checkpoint, rồi so sánh
activation latent với micro-Doppler thủ công.

Dùng:
  python run_trojan.py --ckpt surrogate.pt --config configs/attack.yaml \
      --out target/trojan_m.npy [--steps 200 --batch 64 --device mps]
"""
import os, argparse, yaml
import numpy as np
import torch

from data_utils.feeder import PersonInWiFi3D
from models.factory import build_model
from attack.trojan import optimize_trojan, TrojanTrigger
from attack.trigger import load_trigger


def pick_device(name):
    if name == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    if name == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def grab_clean_batch(cfg, n):
    """Lấy n CSI sạch (đã normalize) từ test split, shape (n,3,180,20)."""
    ds = PersonInWiFi3D('validation', cfg['dataset_root'], cfg['experiment_name'])
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    out = []
    for i in idx:
        raw = ds.load_raw(ds.items[i]['csi'])
        out.append(ds.normalize(raw))
    return np.stack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='surrogate .pt (tu train_backdoor --save)')
    ap.add_argument('--config', default='configs/attack.yaml')
    ap.add_argument('--out', default='target/trojan_m.npy')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--lambda_cm', type=float, default=1.0)
    ap.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'])
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    device = pick_device(a.device)
    print(f'device = {device}')

    # surrogate
    ck = torch.load(a.ckpt, map_location=device)
    model = build_model(cfg['model'], subcarrier_num=ck.get('subcarrier_num', 180)).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    print(f'surrogate loaded: {a.ckpt}')

    csi = grab_clean_batch(cfg, a.batch)
    print(f'clean batch: {csi.shape}')

    trig, info = optimize_trojan(model, csi, device,
                                 eps=cfg['eps'], dose=cfg['dose_max'],
                                 steps=a.steps, lr=a.lr, lambda_cm=a.lambda_cm)
    print('optimize log (act muon TANG, cm muon NHO):')
    for r in info['log']:
        print(f"  step {r['step']:>4}: act={r['act']:.4f}  cm={r['cm']:.5f}")

    np.save(a.out, trig.m)
    print(f'[saved trojan trigger] {a.out}  shape={trig.m.shape}')

    # ---- so sanh activation: trojan vs micro-Doppler thu cong ----
    def mean_act(tg):
        csi_t = np.stack([tg.inject(c.reshape(3, 180, 20).copy()
                          if c.shape == (3, 180, 20) else c, cfg['dose_max'], cfg['eps'])
                          for c in csi])
        # normalize lai nhu feeder roi do activation
        from data_utils.feeder import PersonInWiFi3D as F
        xt = torch.as_tensor(np.stack([F.normalize(x) for x in csi_t]),
                             dtype=torch.float32, device=device)
        with torch.no_grad():
            _, fea = model(xt)
        return float(fea[:, info['target_neurons']].mean())

    md = load_trigger(cfg['action_npy'], top_k=cfg['top_k'],
                      aoa_spread=cfg['aoa_spread'], seed=cfg['seed'])
    with torch.no_grad():
        _, fea_clean = model(torch.as_tensor(csi, dtype=torch.float32, device=device))
    base = float(fea_clean[:, info['target_neurons']].mean())
    print('\n==== ACTIVATION tren target neurons (cao = trigger cong huong manh) ====')
    print(f'  clean        : {base:.4f}')
    print(f'  micro-Doppler: {mean_act(md):.4f}')
    print(f'  Trojan (opt) : {mean_act(trig):.4f}')


if __name__ == '__main__':
    main()
