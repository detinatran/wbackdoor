"""Grid tối ưu Trojan v3 (landed-aware) — thử nhiều (lambda_other, neuron_frac, lambda_land),
đo activation ratio target/other + landed distance, chọn cấu hình tốt nhất.
Tối ưu trigger nhanh (~1 phút/lần) nên thử nhiều, KHÔNG train full mỗi lần.

Dùng: python run_trojan_v3_grid.py --ckpt surrogate200.pt --config configs/attack.yaml --device cuda
"""
import argparse, yaml, json
import numpy as np
import torch
from data_utils.feeder import PersonInWiFi3D
from models.factory import build_model
from attack.trojan import optimize_trojan_v3, TrojanTrigger
from attack.trigger import load_trigger


def grab(cfg, nbatch, bs):
    ds = PersonInWiFi3D('validation', cfg['dataset_root'], cfg['experiment_name'])
    csis, poses = [], []
    for bi in range(nbatch):
        idx = np.linspace(bi, len(ds) - 1, bs).astype(int)
        csis.append(np.stack([ds.normalize(ds.load_raw(ds.items[i]['csi'])) for i in idx]).astype(np.float32))
        poses.append(np.stack([ds.load_pose(ds.items[i]['kpt']) for i in idx]).astype(np.float32))
    return csis, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default='configs/attack.yaml')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--nbatch', type=int, default=4)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--steps', type=int, default=300)
    a = ap.parse_args()
    dev = a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    cfg = yaml.safe_load(open(a.config))

    ck = torch.load(a.ckpt, map_location=dev)
    model = build_model(cfg['model'], subcarrier_num=180).to(dev)
    model.load_state_dict(ck['state_dict']); model.eval()
    csis, poses = grab(cfg, a.nbatch, a.batch)
    print(f'loaded surrogate + {a.nbatch}x{a.batch} samples on {dev}')

    from attack.payload import descendants
    js = descendants(cfg['pivot'])

    # neuron target/other tham chiếu (chọn 1 lần để đo ratio nhất quán)
    from attack.trojan import _limb_relevant_neurons
    tgt0, oth0 = _limb_relevant_neurons(model, torch.as_tensor(csis[0], dtype=torch.float32, device=dev),
                                        dev, cfg['pivot'], k=13)  # ~10%

    def score(trig):
        csi = torch.as_tensor(csis[0], dtype=torch.float32, device=dev)
        xt = np.stack([trig.inject(c, cfg['dose_max'], cfg['eps']) for c in csis[0]])
        from data_utils.feeder import PersonInWiFi3D as F
        xt = torch.as_tensor(np.stack([F.normalize(x) for x in xt]), dtype=torch.float32, device=dev)
        with torch.no_grad():
            _, fea = model(xt)
        at = float(fea[:, tgt0].mean()); ao = float(fea[:, oth0].mean())
        return at, ao, at / (ao + 1e-9)

    GRID = [
        dict(lambda_other=0.5, neuron_frac=0.25, lambda_land=0.0),   # ~ v2 cũ
        dict(lambda_other=1.5, neuron_frac=0.10, lambda_land=0.0),   # khu trú mạnh
        dict(lambda_other=1.5, neuron_frac=0.10, lambda_land=2.0),   # + landed
        dict(lambda_other=2.0, neuron_frac=0.10, lambda_land=3.0),   # khu trú + landed mạnh
        dict(lambda_other=1.0, neuron_frac=0.10, lambda_land=5.0),   # landed trội
    ]
    results = []
    for gi, gp in enumerate(GRID):
        trig, info = optimize_trojan_v3(model, csis, poses, dev, cfg['pivot'],
                                        eps=cfg['eps'], dose=cfg['dose_max'],
                                        theta_max_deg=50.0, steps=a.steps, **gp, seed=0)
        at, ao, ratio = score(trig)
        land_last = info['log'][-1]['land']
        np.save(f'target/trojan_v3_g{gi}.npy', trig.m)
        row = dict(g=gi, **gp, act_t=round(at, 4), act_o=round(ao, 4),
                   ratio=round(ratio, 3), land=round(land_last, 4))
        results.append(row); print(row, flush=True)

    # baseline ref
    for name, npy in [('v1', 'target/trojan_m.npy'), ('v2', 'target/trojan_v2_m.npy')]:
        try:
            at, ao, ratio = score(TrojanTrigger(np.load(npy)))
            print(f'REF {name}: ratio={ratio:.3f} act_t={at:.4f} act_o={ao:.4f}')
        except Exception as e:
            print('skip', name, e)
    json.dump(results, open('trojan_v3_grid.json', 'w'), indent=2)
    best = max(results, key=lambda r: r['ratio'])
    print(f'\nBEST by ratio: g{best["g"]} ratio={best["ratio"]} -> target/trojan_v3_g{best["g"]}.npy')


if __name__ == '__main__':
    main()
