"""Vẽ ground-truth pose 3D người (Person-in-WiFi-3D) — kiểm tra hình người + edges đúng.
Trục z là chiều cao (đứng). Tô đỏ chi đích {pivot + descendants}.

Dùng: python fig_pose_gt.py --sample 5 --out fig_pose_gt.png
"""
import argparse, yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from data_utils.feeder import PersonInWiFi3D
from attack.payload import descendants

# Topology CHÍNH THỨC Person-in-WiFi-3D, lấy từ DT-Pose model.py (generate_adjacency_matrix).
PWIF3D_EDGES = [(0, 1), (0, 2), (2, 5), (3, 0), (4, 2), (5, 7), (6, 3),
                (7, 3), (8, 4), (9, 5), (10, 6), (11, 7), (12, 9), (13, 11)]


def draw(ax, P, limb, title):
    # z = chiều cao -> để z làm trục đứng (mpl trục thứ 3). Hoán: plot (x, y, z) nhưng
    # đặt z lên trục dọc bằng cách dùng z làm tham số thứ 3 (mặc định đã đứng).
    for a, b in PWIF3D_EDGES:
        c = 'crimson' if (a in limb or b in limb) else 'steelblue'
        lw = 3 if c == 'crimson' else 1.6
        ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], [P[a, 2], P[b, 2]], color=c, lw=lw)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='k', s=18)
    for i in range(len(P)):
        ax.text(P[i, 0], P[i, 1], P[i, 2], str(i), fontsize=7, color='dimgray')
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y (depth)'); ax.set_zlabel('z (height)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/attack.yaml')
    ap.add_argument('--sample', type=int, default=5)
    ap.add_argument('--pivot', type=int, default=6)
    ap.add_argument('--out', default='fig_pose_gt.png')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ds = PersonInWiFi3D('validation', cfg['dataset_root'], cfg['experiment_name'])
    P = ds.load_pose(ds.items[a.sample]['kpt'])[0]    # (14,3) ground truth
    limb = descendants(a.pivot)
    print('sample', a.sample, '| chi đích joints', limb)

    fig = plt.figure(figsize=(14, 5))
    for k, (elev, azim, name) in enumerate([(10, -75, 'view A'), (10, -160, 'view B (side)'),
                                            (88, -90, 'view C (top)')]):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        draw(ax, P, limb, name)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((np.ptp(P[:, 0]), np.ptp(P[:, 1]), np.ptp(P[:, 2])))
    plt.suptitle(f'Ground-truth 3D pose (sample {a.sample}) — red = target limb {limb}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(a.out, dpi=150); plt.close()
    print(f'[saved] {a.out}')


if __name__ == '__main__':
    main()
