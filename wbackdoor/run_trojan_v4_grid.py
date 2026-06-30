"""Grid Trojan v4 (landed + leak-penalty) — quét (lambda_land, lambda_leak),
đo land (trúng đích, thấp tốt) + leak-proxy (drift non-target, thấp tốt). Chọn tốt nhất.

Dùng: python run_trojan_v4_grid.py --ckpt surrogate200.pt --config configs/attack.yaml --device cuda
"""
import argparse, yaml, json
import numpy as np
import torch
from data_utils.feeder import PersonInWiFi3D
from models.factory import build_model
from attack.trojan import optimize_trojan_v4, TrojanTrigger
from attack.payload import descendants


def grab(cfg, nbatch, bs):
    ds = PersonInWiFi3D('validation', cfg['dataset_root'], cfg['experiment_name'])
    cs, ps = [], []
    for bi in range(nbatch):
        idx = np.linspace(bi, len(ds) - 1, bs).astype(int)
        cs.append(np.stack([ds.normalize(ds.load_raw(ds.items[i]['csi'])) for i in idx]).astype(np.float32))
        ps.append(np.stack([ds.load_pose(ds.items[i]['kpt']) for i in idx]).astype(np.float32))
    return cs, ps, ds


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
    cs, ps, ds = grab(cfg, a.nbatch, a.batch)
    js = descendants(cfg['pivot']); other_j = [j for j in range(14) if j not in js]
    print(f'loaded on {dev}, {a.nbatch}x{a.batch} samples')

    # đo leak thật trên data: drift non-target joint giữa pred-trigger và pred-clean
    def measure(trig):
        from data_utils.feeder import PersonInWiFi3D as F
        with torch.no_grad():
            xc = torch.as_tensor(cs[0], dtype=torch.float32, device=dev)
            pc = model(xc)[0][:, 0]
            xt = np.stack([trig.inject(c, cfg['dose_max'], cfg['eps']) for c in cs[0]])
            xt = torch.as_tensor(np.stack([F.normalize(x) for x in xt]), dtype=torch.float32, device=dev)
            pt = model(xt)[0][:, 0]
        leak = (pt[:, other_j] - pc[:, other_j]).norm(dim=-1).mean().item() * 1000
        disp = (pt[:, js] - pc[:, js]).norm(dim=-1).mean().item() * 1000
        return leak, disp

    GRID = [
        dict(lambda_land=5.0, lambda_leak=0.0),    # = g4 (đối chứng)
        dict(lambda_land=5.0, lambda_leak=3.0),
        dict(lambda_land=5.0, lambda_leak=6.0),
        dict(lambda_land=8.0, lambda_leak=6.0),
        dict(lambda_land=8.0, lambda_leak=10.0),
    ]
    results = []
    for gi, gp in enumerate(GRID):
        trig, info = optimize_trojan_v4(model, cs, ps, dev, cfg['pivot'],
                                        eps=cfg['eps'], dose=cfg['dose_max'], theta_max_deg=50.0,
                                        steps=a.steps, lambda_other=1.0, neuron_frac=0.1, **gp, seed=0)
        leak, disp = measure(trig)
        np.save(f'target/trojan_v4_g{gi}.npy', trig.m)
        row = dict(g=gi, **gp, leak_mm=round(leak, 1), disp_mm=round(disp, 1),
                   land_last=round(info['log'][-1]['land'], 4))
        results.append(row); print(row, flush=True)

    # ref leak cua v2,g4,microDoppler de so
    print('\n--- tham chieu leak (mm) tren cung batch ---')
    for name, npy in [('v2', 'target/trojan_v2_m.npy'), ('v4_g4base', None)]:
        if npy:
            try:
                leak, disp = measure(TrojanTrigger(np.load(npy))); print(f'  {name}: leak={leak:.0f} disp={disp:.0f}')
            except Exception as e: print(' skip', name, e)
    json.dump(results, open('trojan_v4_grid.json', 'w'), indent=2)
    # chọn: leak thấp + disp còn đủ lớn (payload mạnh). Ưu tiên leak thấp với disp>200mm
    ok = [r for r in results if r['disp_mm'] > 150]
    best = min(ok or results, key=lambda r: r['leak_mm'])
    print(f'\nBEST (leak thấp, disp đủ): g{best["g"]} leak={best["leak_mm"]}mm disp={best["disp_mm"]}mm '
          f'-> target/trojan_v4_g{best["g"]}.npy')


if __name__ == '__main__':
    main()
