"""Trojan-only figures: dose-response, attack metrics, leak/plausibility, trigger pattern."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM = 1000.0
C = 'tab:red'
tj = json.load(open('results_cmp_trojan.json'))
grid = np.array(tj['dose_grid'])
disp = np.array(tj['displacement']) * MM
tmp = np.array(tj['tmpjpe']) * MM
leak = np.array(tj['nontarget_mpjpe']) * MM
plaus = np.array(tj['plausibility'])
dr = tj['dose_response']
asr = tj['asr@ref']

# ============ FIG 1: dose-response ============
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(grid, disp, 'o-', color=C, lw=2.2, ms=7, label='Target-limb displacement')
ax.plot(grid, tmp, 's--', color='darkorange', lw=2, ms=6, label='t-MPJPE (dist. to target)')
coef = np.polyfit(grid, disp, 1)
ax.plot(grid, np.polyval(coef, grid), ':', color='gray', lw=1.5,
        label=f'Linear fit (slope={dr["slope"]:.3f})')
ax.set_xlabel('Dose (trigger intensity)'); ax.set_ylabel('mm')
ax.set_title(f'Trojan dose-response  (Spearman={dr["spearman"]:.2f}, '
             f'$r^2_{{ramp}}$={dr["r2_ramp"]:.2f})')
ax.legend(); ax.grid(alpha=0.3)
ax.text(0.02, 0.97, 'Analog: payload scales continuously with dose',
        transform=ax.transAxes, va='top', fontsize=9, color=C)
plt.tight_layout(); plt.savefig('fig_trojan_1_doseresponse.png', dpi=150); plt.close()

# ============ FIG 2: attack metrics ============
fig, ax = plt.subplots(figsize=(7, 5))
names = ['ASR', 'frac_landed', 'frac_preserved', 'clean PCK@0.5', 'Spearman']
vals = [asr['asr'], asr['frac_landed'], asr['frac_preserved'],
        tj['clean_pck@0.5'], dr['spearman']]
bars = ax.bar(names, vals, color=['crimson', 'crimson', 'teal', 'teal', 'slateblue'])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_ylim(0, 1.08); ax.set_ylabel('Value (0-1)')
ax.set_title('Trojan - attack metrics ($\\theta$=50$\\degree$, $\\rho$=0.1, 200 epochs)')
ax.axhline(0, color='k', lw=0.5)
plt.xticks(rotation=15); plt.tight_layout()
plt.savefig('fig_trojan_2_metrics.png', dpi=150); plt.close()

# ============ FIG 3: leak vs plausibility ============
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(grid, leak, 'o-', color='tab:purple', lw=2, label='Non-target leak (mm)')
ax1.set_xlabel('Dose'); ax1.set_ylabel('Non-target leak (mm)', color='tab:purple')
ax1.tick_params(axis='y', labelcolor='tab:purple')
ax2 = ax1.twinx()
ax2.plot(grid, plaus, 's--', color=C, lw=2, label='Plausibility error')
ax2.set_ylabel('Plausibility error (bone-length dev.)', color=C)
ax2.tick_params(axis='y', labelcolor=C)
ax1.set_title('Trojan - leak & plausibility error grow with dose')
ax1.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_trojan_3_leak_plaus.png', dpi=150); plt.close()

# ============ FIG 4: trigger pattern m ============
m = np.load('target/trojan_m.npy')
amp = np.abs(m); pha = np.angle(m)
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for a in range(3):
    im = axs[0, a].imshow(amp[a], aspect='auto', origin='lower', cmap='viridis')
    axs[0, a].set_title(f'|m| antenna {a}'); axs[0, a].set_xlabel('Packet'); axs[0, a].set_ylabel('Subcarrier')
    fig.colorbar(im, ax=axs[0, a], fraction=0.046)
    im2 = axs[1, a].imshow(pha[a], aspect='auto', origin='lower', cmap='twilight')
    axs[1, a].set_title(f'$\\angle$m antenna {a}'); axs[1, a].set_xlabel('Packet'); axs[1, a].set_ylabel('Subcarrier')
    fig.colorbar(im2, ax=axs[1, a], fraction=0.046)
cm = np.abs(m.mean(0)).mean()
diff = np.abs(m - m.mean(0, keepdims=True)).mean()
plt.suptitle(f'Trojan trigger pattern m (3,30,20) - antenna-differential '
             f'(diff={diff:.3f} >> common-mode={cm:.3f})', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('fig_trojan_4_pattern.png', dpi=150); plt.close()

print('saved 4 figures (English labels)')
