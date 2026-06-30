"""Lưới 3x4 pose 3D theo style báo cáo: mỗi panel so GT vs Clean-pred vs Triggered-pred.
- Gray solid  : Ground Truth
- Teal dashed : model dự đoán khi SẠCH (no trigger)
- Red dotted  : model dự đoán khi CÓ TRIGGER (dose=1.0)
Lọc: chỉ vẽ sample có clean-MPJPE < ngưỡng (chọn frame model dự đoán tốt).

Dùng:
  python fig_pose_grid.py --ckpt cmp_trojan_v2.pt --config configs/_tsne_v2.yaml \
      --out fig_grid_v2.png --title "Trojan v2" --mpjpe_max 0.4 --dose 1.0
"""
import argparse, yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from data_utils.feeder import PersonInWiFi3D
from models.factory import build_model
from train_backdoor import build_trigger, _pick_device

# Topology CHÍNH THỨC Person-in-WiFi-3D (DT-Pose model.py).
EDGES = [(0, 1), (0, 2), (2, 5), (3, 0), (4, 2), (5, 7), (6, 3),
         (7, 3), (8, 4), (9, 5), (10, 6), (11, 7), (12, 9), (13, 11)]


def _flipz(P):
    # Trục z của dataset hướng XUỐNG (z lớn = thấp). Lật để hiển thị người đứng đúng.
    Q = P.copy(); Q[:, 2] = -Q[:, 2]; return Q


def draw(ax, P, color, ls, lw=1.6, marker=None):
    P = _flipz(P)
    for a, b in EDGES:
        ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], [P[a, 2], P[b, 2]],
                color=color, ls=ls, lw=lw)
    if marker:
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=color, s=8, marker=marker)


def mpjpe(a, b):
    return np.linalg.norm(a - b, axis=-1).mean() * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default='fig_grid.png')
    ap.add_argument('--title', default='model')
    ap.add_argument('--dose', type=float, default=1.0)
    ap.add_argument('--mpjpe_max', type=float, default=0.4, help='lọc clean-MPJPE (m)')
    ap.add_argument('--rows', type=int, default=4)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--scan', type=int, default=200, help='số sample quét để lọc')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    dev = cfg.get('device') or _pick_device()
    ck = torch.load(a.ckpt, map_location=dev)
    model = build_model(cfg['model'], subcarrier_num=180).to(dev)
    model.load_state_dict(ck['state_dict']); model.eval()
    trig = build_trigger(cfg)
    ds = PersonInWiFi3D('validation', cfg['dataset_root'], cfg['experiment_name'])

    need = a.rows * a.cols
    picks = []
    idxs = np.linspace(0, len(ds) - 1, a.scan).astype(int)
    for i in idxs:
        raw = ds.load_raw(ds.items[i]['csi'])
        gt = ds.load_pose(ds.items[i]['kpt'])[0]
        xc = torch.as_tensor(ds.normalize(raw)[None], dtype=torch.float32, device=dev)
        with torch.no_grad():
            pc, _ = model(xc)
        pc = pc[0, 0].cpu().numpy()
        m = mpjpe(pc, gt)
        if m < a.mpjpe_max * 1000:
            xt = torch.as_tensor(ds.normalize(trig.inject(raw, a.dose, cfg['eps']))[None],
                                 dtype=torch.float32, device=dev)
            with torch.no_grad():
                pt, _ = model(xt)
            pt = pt[0, 0].cpu().numpy()
            picks.append((ds.items[i]['name'], gt, pc, pt, m, mpjpe(pt, gt)))
        if len(picks) >= need:
            break
    print(f'chọn {len(picks)}/{need} sample (clean-MPJPE < {a.mpjpe_max*1000:.0f}mm)')

    fig = plt.figure(figsize=(4.5 * a.cols, 4.2 * a.rows))
    for k, (name, gt, pc, pt, mc, mt) in enumerate(picks):
        ax = fig.add_subplot(a.rows, a.cols, k + 1, projection='3d')
        draw(ax, gt, '0.4', '-', 2.0)                 # GT xám solid
        draw(ax, pc, 'teal', '--', 1.6)               # clean teal dashed
        draw(ax, pt, 'crimson', ':', 1.8, marker='.')  # triggered đỏ dotted
        ax.set_title(f'{name}\nclean MPJPE {mc:.0f} | trig {mt:.0f} mm', fontsize=8)
        ax.view_init(elev=5, azim=-90)
        allp = np.concatenate([gt, pc, pt])
        ax.set_box_aspect((np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), np.ptp(allp[:, 2])))
        ax.tick_params(labelsize=6)
    plt.suptitle(f'3D pose: GT (gray) vs clean-pred (teal) vs triggered-pred (red, dose={a.dose}) — {a.title}',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(a.out, dpi=140); plt.close()
    print(f'[saved] {a.out}')


if __name__ == '__main__':
    main()
