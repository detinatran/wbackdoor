"""Tối ưu Trojan v2 (localized + multi-batch + norm-budget) trên Mac (MPS).
Maximize neuron LÁI chi đích, phạt neuron khác -> Trojan mạnh NHƯNG ít leak.

Dùng:
  python run_trojan_v2.py --ckpt surrogate.pt --config configs/attack.yaml \
      --out target/trojan_v2_m.npy --nbatch 4 --batch 48 --steps 300 \
      --lambda_other 0.5 --norm_budget 40
"""
import argparse, yaml
import numpy as np
import torch

from data_utils.feeder import PersonInWiFi3D
from models.factory import build_model
from attack.trojan import optimize_trojan_v2, TrojanTrigger
from attack.trigger import load_trigger


def pick_device(name):
    if name == 'mps' and torch.backends.mps.is_available(): return 'mps'
    if name == 'cuda' and torch.cuda.is_available(): return 'cuda'
    return 'cpu'


def grab_batches(cfg, nbatch, bs):
    ds = PersonInWiFi3D('validation', cfg['dataset_root'], cfg['experiment_name'])
    out = []
    for bi in range(nbatch):
        idx = np.linspace(bi, len(ds) - 1, bs).astype(int)
        out.append(np.stack([ds.normalize(ds.load_raw(ds.items[i]['csi'])) for i in idx]).astype(np.float32))
    return out, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default='configs/attack.yaml')
    ap.add_argument('--out', default='target/trojan_v2_m.npy')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--nbatch', type=int, default=4)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--lambda_other', type=float, default=0.5)
    ap.add_argument('--lambda_cm', type=float, default=1.0)
    ap.add_argument('--norm_budget', type=float, default=None)
    ap.add_argument('--device', default='mps')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    device = pick_device(a.device); print('device =', device)
    ck = torch.load(a.ckpt, map_location=device)
    model = build_model(cfg['model'], subcarrier_num=ck.get('subcarrier_num', 180)).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    print('surrogate loaded')

    batches, ds = grab_batches(cfg, a.nbatch, a.batch)
    print(f'{a.nbatch} batches x {a.batch} samples')

    trig, info = optimize_trojan_v2(model, batches, device, pivot=cfg['pivot'],
                                    eps=cfg['eps'], dose=cfg['dose_max'], steps=a.steps,
                                    lr=a.lr, lambda_other=a.lambda_other, lambda_cm=a.lambda_cm,
                                    norm_budget=a.norm_budget)
    print('optimize log (act_target TANG, act_other GIAM, cm NHO):')
    for r in info['log']:
        print(f"  step {r['step']:>4}: target={r['act_target']:.4f} "
              f"other={r['act_other']:.4f} cm={r['cm']:.5f}")
    np.save(a.out, trig.m)
    print(f'[saved] {a.out}')

    # so activation target vs other: trojan v1, v2, micro-doppler
    def acts(tg):
        csi = batches[0]
        xt = np.stack([tg.inject(c, cfg['dose_max'], cfg['eps']) for c in csi])
        from data_utils.feeder import PersonInWiFi3D as F
        xt = torch.as_tensor(np.stack([F.normalize(x) for x in xt]),
                             dtype=torch.float32, device=device)
        with torch.no_grad(): _, fea = model(xt)
        return (float(fea[:, info['target_neurons']].mean()),
                float(fea[:, info['other_neurons']].mean()))

    md = load_trigger(cfg['action_npy'], top_k=cfg['top_k'], aoa_spread=cfg['aoa_spread'], seed=cfg['seed'])
    rows = [('micro-Doppler', md), ('Trojan v2 (localized)', trig)]
    try:
        v1 = TrojanTrigger(np.load('target/trojan_m.npy')); rows.insert(1, ('Trojan v1', v1))
    except Exception:
        pass
    print('\n==== activation: target-neurons (cao=tot) | other-neurons (thap=it leak) ====')
    for name, tg in rows:
        at, ao = acts(tg)
        print(f'  {name:<22}: target={at:.4f}  other={ao:.4f}  ratio={at/(ao+1e-9):.2f}')


if __name__ == '__main__':
    main()
