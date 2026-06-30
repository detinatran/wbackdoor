"""Vẽ figure so sánh Trojan vs micro-Doppler từ results_cmp_*.json (đơn vị mm)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM = 1000.0
md = json.load(open('results_cmp_microdoppler.json'))
tj = json.load(open('results_cmp_trojan.json'))
grid = md['dose_grid']

C_MD, C_TJ = 'tab:blue', 'tab:red'

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# ---- (1) dose-response: displacement vs dose ----
ax = axs[0, 0]
ax.plot(grid, [x * MM for x in md['displacement']], 'o-', color=C_MD, lw=2,
        label=f"micro-Doppler (slope={md['dose_response']['slope']:.3f}, ρ_s={md['dose_response']['spearman']:.2f})")
ax.plot(grid, [x * MM for x in tj['displacement']], 's-', color=C_TJ, lw=2,
        label=f"Trojan (slope={tj['dose_response']['slope']:.3f}, ρ_s={tj['dose_response']['spearman']:.2f})")
ax.set_xlabel('dose'); ax.set_ylabel('target-limb displacement (mm)')
ax.set_title('(a) Dose–response: payload vs dose'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- (2) bar: các chỉ số attack chính ----
ax = axs[0, 1]
labels = ['ASR', 'frac_landed', 'disp@max\n(/100mm)', 'slope\n(×1)']
mvals = [md['asr@ref']['asr'], md['asr@ref']['frac_landed'],
         md['displacement'][-1] * MM / 100, md['dose_response']['slope']]
tvals = [tj['asr@ref']['asr'], tj['asr@ref']['frac_landed'],
         tj['displacement'][-1] * MM / 100, tj['dose_response']['slope']]
x = np.arange(len(labels)); w = 0.36
ax.bar(x - w/2, mvals, w, color=C_MD, label='micro-Doppler')
ax.bar(x + w/2, tvals, w, color=C_TJ, label='Trojan')
for i, (a, b) in enumerate(zip(mvals, tvals)):
    ax.text(i - w/2, a, f'{a:.2f}', ha='center', va='bottom', fontsize=7)
    ax.text(i + w/2, b, f'{b:.2f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_title('(b) Sức mạnh tấn công (cao = mạnh hơn)'); ax.legend(fontsize=8)

# ---- (3) localization trade-off: leak + preservation ----
ax = axs[1, 0]
labels2 = ['non-target leak\n(mm)', 'frac_preserved\n(×100)', 'clean_pck@0.5\n(×100)']
mv2 = [md['nontarget_mpjpe'][-1] * MM, md['asr@ref']['frac_preserved'] * 100, md['clean_pck@0.5'] * 100]
tv2 = [tj['nontarget_mpjpe'][-1] * MM, tj['asr@ref']['frac_preserved'] * 100, tj['clean_pck@0.5'] * 100]
x2 = np.arange(len(labels2))
ax.bar(x2 - w/2, mv2, w, color=C_MD, label='micro-Doppler')
ax.bar(x2 + w/2, tv2, w, color=C_TJ, label='Trojan')
for i, (a, b) in enumerate(zip(mv2, tv2)):
    ax.text(i - w/2, a, f'{a:.1f}', ha='center', va='bottom', fontsize=7)
    ax.text(i + w/2, b, f'{b:.1f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x2); ax.set_xticklabels(labels2, fontsize=8)
ax.set_title('(c) Khu trú & tàng hình (leak thấp=tốt, preserved/pck cao=tốt)'); ax.legend(fontsize=8)

# ---- (4) plausibility vs dose ----
ax = axs[1, 1]
ax.plot(grid, md['plausibility'], 'o-', color=C_MD, lw=2, label='micro-Doppler')
ax.plot(grid, tj['plausibility'], 's-', color=C_TJ, lw=2, label='Trojan')
ax.set_xlabel('dose'); ax.set_ylabel('plausibility error (bone-length dev.)')
ax.set_title('(d) Plausibility theo dose (thấp = pose hợp lý hơn)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle('Trojan (optimized) vs micro-Doppler (hand-crafted) — θ=50°, ρ=0.1, 200 epoch',
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('fig_trojan_compare.png', dpi=150); plt.close()
print('saved fig_trojan_compare.png')
