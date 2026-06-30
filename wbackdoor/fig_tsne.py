"""t-SNE của latent `fea` cho một backdoored model + trigger.
Chiếu fea của: (clean) vs (trigger ở các mức dose) -> cho thấy trigger tách cluster
trong latent space, và payload mạnh dần theo dose (cluster dịch dần).

Dùng:
  python fig_tsne.py --ckpt cmp_v3_g4.pt --config configs/_cmp_v3_g4.yaml \
      --out fig_tsne_v3_g4.png --title "Trojan v3 (g4)" --n 300
"""
import argparse, yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data_utils.feeder import PersonInWiFi3D
from models.factory import build_model
from train_backdoor import build_trigger, _pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default='fig_tsne.png')
    ap.add_argument('--title', default='model')
    ap.add_argument('--n', type=int, default=300, help='số mẫu mỗi nhóm')
    ap.add_argument('--doses', type=float, nargs='+', default=[0.0, 1.0])
    ap.add_argument('--mode', default='binary', choices=['binary', 'dose'],
                    help='binary=clean vs triggered; dose=nhiều mức dose')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    dev = cfg.get('device') or _pick_device()
    ck = torch.load(a.ckpt, map_location=dev)
    model = build_model(cfg['model'], subcarrier_num=180).to(dev)
    model.load_state_dict(ck['state_dict']); model.eval()
    trig = build_trigger(cfg)

    ds = PersonInWiFi3D('validation', cfg['dataset_root'], cfg['experiment_name'])
    idx = np.linspace(0, len(ds) - 1, a.n).astype(int)
    raws = [ds.load_raw(ds.items[i]['csi']) for i in idx]

    feats, labels = [], []
    for d in a.doses:
        xs = []
        for raw in raws:
            csi = trig.inject(raw, d, cfg['eps']) if d > 0 else raw
            xs.append(ds.normalize(csi))
        x = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=dev)
        with torch.no_grad():
            _, fea = model(x)                       # (n,128)
        feats.append(fea.cpu().numpy()); labels += [d] * len(idx)
    F = np.concatenate(feats); labels = np.array(labels)

    print(f'fea {F.shape}, running t-SNE...')
    emb = TSNE(n_components=2, perplexity=30, init='pca', random_state=0).fit_transform(F)

    plt.figure(figsize=(7, 6))
    if a.mode == 'binary':
        # clean (xám) vs triggered (đỏ) — đo độ tách bằng silhouette
        from sklearn.metrics import silhouette_score
        bin_lab = (labels > 0).astype(int)
        sil = silhouette_score(emb, bin_lab)
        for v, c, lab in [(0, '0.6', 'clean'), (1, 'crimson', 'triggered (dose=1.0)')]:
            mask = bin_lab == v
            plt.scatter(emb[mask, 0], emb[mask, 1], s=16, alpha=0.6, color=c,
                        label=lab, edgecolors='none')
        plt.title(f't-SNE of latent fea — {a.title}  (silhouette={sil:.3f})')
    else:
        cmap = plt.cm.viridis
        for d in a.doses:
            mask = labels == d
            lab = 'clean' if d == 0 else f'trigger@dose={d}'
            plt.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=0.7,
                        color='gray' if d == 0 else cmap(d), label=lab, edgecolors='none')
        plt.title(f't-SNE of latent fea — {a.title}')
    plt.legend(fontsize=9, loc='best')
    plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
    plt.tight_layout(); plt.savefig(a.out, dpi=150); plt.close()
    print(f'[saved] {a.out}')


if __name__ == '__main__':
    main()
