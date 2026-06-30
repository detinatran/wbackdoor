"""
Fine-tuning defense (Liu et al., RAID 2018) — đánh giá IRREVERSIBILITY của backdoor.

Quy trình:
  1. Load model ĐÃ NHIỄM backdoor (checkpoint).
  2. Đo ASR/leak/clean ban đầu (trigger tương ứng).
  3. Fine-tune model trên DATA SẠCH (không trigger, pose thật) — đây là defense.
  4. Đo lại ASR sau mỗi vài epoch.
Backdoor IRREVERSIBLE nếu ASR vẫn cao sau fine-tune.

Dùng:
  python defense_finetune.py --ckpt cmp_trojan.pt --config configs/_cmp_trojan.yaml \
      --ft_epochs 30 --ft_lr 1e-4 --out defense_trojan_v1.json
"""
import argparse, json, yaml, copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils.feeder import PersonInWiFi3D
from attack.poison import PoisonedDataset, collate
from models.factory import build_model
from train_backdoor import build_trigger, evaluate, _pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True, help='config khớp trigger của checkpoint')
    ap.add_argument('--ft_epochs', type=int, default=30)
    ap.add_argument('--ft_lr', type=float, default=1e-4)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--out', default='defense_result.json')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    device = cfg.get('device') or _pick_device()
    print(f'device={device}  trigger={cfg.get("trigger_type","microdoppler")}')

    base_train = PersonInWiFi3D('training', cfg['dataset_root'], cfg['experiment_name'])
    base_test = PersonInWiFi3D('validation', cfg['dataset_root'], cfg['experiment_name'])
    trig = build_trigger(cfg)

    model = build_model(cfg['model'], subcarrier_num=180).to(device)
    ck = torch.load(a.ckpt, map_location=device)
    model.load_state_dict(ck['state_dict'])
    print(f'loaded poisoned model: {a.ckpt}')

    def measure(tag):
        res = evaluate(model, base_test, trig, cfg, device)
        row = {'tag': tag,
               'asr': res['asr@ref'].get('asr'),
               'frac_landed': res['asr@ref'].get('frac_landed'),
               'displacement_mm': res['displacement'][-1] * 1000,
               'nontarget_mpjpe_mm': res['nontarget_mpjpe'][-1] * 1000,
               'clean_mpjpe_mm': res['clean_mpjpe'] * 1000,
               'clean_pck': res['clean_pck@0.5']}
        print(f"  [{tag}] ASR={row['asr']:.3f} disp={row['displacement_mm']:.0f}mm "
              f"clean_pck={row['clean_pck']:.3f}")
        return row

    history = []
    print('\n=== ASR truoc defense ===')
    history.append(measure('before'))

    # ----- fine-tune defense: train tren DATA SACH (rho=0) -----
    clean_ds = PoisonedDataset(base_train, trig, mode='train', rho=0.0, pivot=cfg['pivot'])
    loader = DataLoader(clean_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=a.ft_lr)
    print(f'\n=== fine-tune {a.ft_epochs} epoch tren data sach (lr={a.ft_lr}) ===')
    for ep in range(1, a.ft_epochs + 1):
        model.train(); losses = []
        for b in loader:
            csi, pose = b['csi'].to(device), b['pose'].to(device)
            pred, _ = model(csi)
            loss = torch.mean(torch.norm(pred - pose, dim=-1))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % a.eval_every == 0 or ep == a.ft_epochs:
            print(f'ft epoch {ep}: loss={np.mean(losses):.4f}')
            history.append(measure(f'ft_ep{ep}'))

    json.dump({'ckpt': a.ckpt, 'trigger': cfg.get('trigger_type', 'microdoppler'),
               'ft_epochs': a.ft_epochs, 'ft_lr': a.ft_lr, 'history': history},
              open(a.out, 'w'), indent=2)
    print(f'\n[saved] {a.out}')
    asr0, asr1 = history[0]['asr'], history[-1]['asr']
    drop = (asr0 - asr1) / (asr0 + 1e-9) * 100
    print(f'\n==== KET QUA: ASR {asr0:.3f} -> {asr1:.3f}  (giam {drop:.0f}%) ====')
    print('IRREVERSIBLE' if asr1 > 0.5 * asr0 else 'backdoor bi go phan lon')


if __name__ == '__main__':
    main()
