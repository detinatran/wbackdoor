"""Gộp các res_*.json (từ sweep_parallel.sh) -> bảng csv/json + biểu đồ giống sweep.py."""
import os, sys, glob, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM = 1000.0
OUT = sys.argv[1] if len(sys.argv) > 1 else 'sweep_out_full'

rows = []
for f in sorted(glob.glob(os.path.join(OUT, 'res_*.json'))):
    r = json.load(open(f))
    dr = r['dose_response']
    asr = r['asr@ref']
    row = {
        'theta_max_deg': r['_theta_max_deg'], 'rho': r['_rho'],
        'clean_mpjpe_mm': r['clean_mpjpe'] * MM,
        'displacement_mm': r['displacement'][-1] * MM,
        'nontarget_mpjpe_mm': r['nontarget_mpjpe'][-1] * MM,
        'plausibility': r['plausibility'][-1],
        'spearman': dr['spearman'],
        'ramp_minus_step': dr['ramp_minus_step'],
    }
    # metric mới (calibrated ASR) nếu có; fallback field cũ frac_moved
    if 'tmpjpe' in r:
        row['tmpjpe_mm'] = r['tmpjpe'][-1] * MM
    if 'clean_target_floor' in r:
        row['clean_target_floor_mm'] = r['clean_target_floor'] * MM
    if 'asr' in asr:                      # bản mới: attack_metrics
        row['asr'] = asr['asr']
        row['frac_landed'] = asr.get('frac_landed')
        row['frac_preserved'] = asr.get('frac_preserved')
        row['_payload_key'] = asr['asr']            # dùng để tô đậm/chọn best
    else:                                 # bản cũ
        row['frac_moved'] = asr.get('frac_moved')
        row['_payload_key'] = asr.get('frac_moved')
    rows.append(row)
if not rows:
    print('Không thấy res_*.json trong', OUT); sys.exit(1)
print(f'Gộp {len(rows)} kết quả.')

json.dump(rows, open(os.path.join(OUT, 'sweep_results.json'), 'w'), indent=2)
with open(os.path.join(OUT, 'sweep_results.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# fig tradeoff -------------------------------------------------------------
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
rhos = sorted({r['rho'] for r in rows})
colors = {rho: c for rho, c in zip(rhos, ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])}
clean = np.mean([r['clean_mpjpe_mm'] for r in rows])

def _scatter(ax, label_thetas=True, spread=True):
    rho_hi = max(rhos)
    for rho in rhos:
        pts = sorted([r for r in rows if r['rho'] == rho], key=lambda r: r['displacement_mm'])
        xs = [p['displacement_mm'] for p in pts]; ys = [p['nontarget_mpjpe_mm'] for p in pts]
        hi = (rho == rho_hi)
        # rho=0.1 (đường có ý nghĩa) đậm & nổi lên trên; rho nhỏ mờ & chìm xuống
        ax.plot(xs, ys, 'o-', color=colors[rho], label=f'rho={rho}',
                ms=6 if hi else 4, lw=2.5 if hi else 1.5,
                alpha=1.0 if hi else 0.55, zorder=6 if hi else 2)
        if label_thetas:
            for i, p in enumerate(pts):
                ang = (i / max(1, len(pts) - 1)) * 2 - 1
                ax.annotate(f"{int(p['theta_max_deg'])}",
                            (p['displacement_mm'], p['nontarget_mpjpe_mm']),
                            fontsize=7, color=colors[rho],
                            alpha=1.0 if hi else 0.6, zorder=7 if hi else 3,
                            xytext=(8, 10 * ang), textcoords='offset points',
                            arrowprops=dict(arrowstyle='-', lw=0.4, color='0.6'))

fig, ax = plt.subplots(figsize=(7.5, 5.5))
_scatter(ax)
ax.axhline(clean, ls='--', c='gray', lw=1)
ax.text(ax.get_xlim()[1], clean, f' clean MPJPE ~{clean:.0f}mm',
        va='bottom', ha='right', fontsize=8, c='gray')
# best trade-off: rho cao nhất, displacement lớn mà leak nhỏ (cân bằng)
act = [r for r in rows if r['rho'] == max(rhos)]
best = min(act, key=lambda r: r['nontarget_mpjpe_mm'] - 0.05 * r['displacement_mm'])
# max payload: rho cao nhất, displacement lớn nhất
mxp = max(act, key=lambda r: r['displacement_mm'])
ax.scatter([best['displacement_mm']], [best['nontarget_mpjpe_mm']],
           s=210, facecolors='none', edgecolors='crimson', lw=2.0, zorder=8)
ax.annotate(f"best trade-off: $\\theta$={int(best['theta_max_deg'])}, $\\rho$={best['rho']}",
            (best['displacement_mm'], best['nontarget_mpjpe_mm']),
            xytext=(-16, 26), textcoords='offset points', fontsize=8.5, color='crimson',
            ha='right', arrowprops=dict(arrowstyle='->', color='crimson', lw=1.1))
if mxp is not best:
    ax.annotate(f"max payload: $\\theta$={int(mxp['theta_max_deg'])}",
                (mxp['displacement_mm'], mxp['nontarget_mpjpe_mm']),
                xytext=(6, 20), textcoords='offset points', fontsize=8, color='0.3',
                ha='left', arrowprops=dict(arrowstyle='->', color='0.5', lw=0.9))
ax.margins(y=0.18)   # chừa khoảng trống đáy/đỉnh để nhãn không đè trục

# inset: zoom vào cụm điểm "chưa kích hoạt" (displacement nhỏ)
lowx = [r['displacement_mm'] for r in rows if r['displacement_mm'] < 50]
lowy = [r['nontarget_mpjpe_mm'] for r in rows if r['displacement_mm'] < 50]
if len(lowx) > 3:
    # đặt inset góc trên-trái (vùng trống), không che đường đỏ ở dưới
    axins = inset_axes(ax, width='40%', height='40%', loc='upper center', borderpad=1.5)
    _scatter(axins, label_thetas=True)
    axins.set_xlim(min(lowx) - 2, max(lowx) + 5); axins.set_ylim(min(lowy) - 1, max(lowy) + 2.5)
    axins.set_title('zoom: rho<=0.05 (chua kich hoat)', fontsize=7)
    axins.tick_params(labelsize=6); axins.get_legend() and axins.get_legend().remove()
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.65', lw=0.8, alpha=0.5)

ax.set_xlabel('target-limb displacement (mm)   [bigger = stronger payload]')
ax.set_ylabel('non-target leak MPJPE (mm)   [smaller = better localization]')
ax.set_title('Localization vs payload tradeoff (label = theta_max deg)')
ax.legend(loc='upper left', fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_tradeoff.png'), dpi=150); plt.close()

# heatmaps
thetas = sorted({r['theta_max_deg'] for r in rows})
def grid(key):
    M = np.full((len(rhos), len(thetas)), np.nan)
    for r in rows:
        M[rhos.index(r['rho']), thetas.index(r['theta_max_deg'])] = r[key]
    return M
panels = [('nontarget_mpjpe_mm', 'leak (mm) lower=better'),
          ('displacement_mm', 'displacement (mm)'),
          ('ramp_minus_step', 'ramp - step R2 (>0=analog)')]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (key, title) in zip(axs, panels):
    M = grid(key)
    im = ax.imshow(M, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(range(len(thetas))); ax.set_xticklabels([int(t) for t in thetas])
    ax.set_yticks(range(len(rhos))); ax.set_yticklabels(rhos)
    ax.set_xlabel('theta_max (deg)'); ax.set_ylabel('rho'); ax.set_title(title)
    for a in range(len(rhos)):
        for b in range(len(thetas)):
            ax.text(b, a, f'{M[a, b]:.2f}', ha='center', va='center', color='w', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig_heatmaps.png'), dpi=150); plt.close()
print(f'Xong -> {OUT}/sweep_results.{{json,csv}}, fig_tradeoff.png, fig_heatmaps.png')
