"""Tạo bộ hình cho slide từ kết quả thật (English labels)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM = 1000.0
def load(f): return json.load(open(f))

md = load('results_cmp_microdoppler.json')
v2 = load('results_cmp_trojan.json')
v4 = load('results_cmp_v4_g3.json')
wa = load('results_cmp_wanet.json')
ft = load('results_cmp_ftrojan.json')

METHODS = [('FTrojan', ft, 'tab:gray'), ('Trojan v2', v2, 'tab:orange'),
           ('Trojan v4', v4, 'tab:red'), ('WaNet', wa, 'tab:green')]
def asr(d): return d['asr@ref']['asr']
def leak(d): return d['nontarget_mpjpe'][-1]*MM
def disp(d): return d['displacement'][-1]*MM
def presv(d): return d['asr@ref']['frac_preserved']

os.makedirs('figs', exist_ok=True)

# ---------- FIG A: bar so sánh ASR + leak (5 method, không micro-Doppler) ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
names = [m[0] for m in METHODS]
colors = [m[2] for m in METHODS]
a1.bar(names, [asr(m[1]) for m in METHODS], color=colors)
for i, m in enumerate(METHODS): a1.text(i, asr(m[1]), f'{asr(m[1]):.3f}', ha='center', va='bottom')
a1.set_title('Attack Success Rate (higher = stronger)'); a1.set_ylabel('ASR'); a1.set_ylim(0, 0.6)
a2.bar(names, [leak(m[1]) for m in METHODS], color=colors)
for i, m in enumerate(METHODS): a2.text(i, leak(m[1]), f'{leak(m[1]):.0f}', ha='center', va='bottom')
a2.axhline(leak(md), ls='--', c='purple', lw=1.5)
a2.text(3.4, leak(md), f' micro-Doppler ref ({leak(md):.0f}mm)', va='bottom', ha='right', color='purple', fontsize=8)
a2.set_title('Non-target leak (lower = better localization)'); a2.set_ylabel('leak (mm)')
plt.tight_layout(); plt.savefig('figs/fig_compare_bar.png', dpi=150); plt.close()

# ---------- FIG B: dose-response 3 method (analog) ----------
grid = np.array(v2['dose_grid'])
plt.figure(figsize=(7, 5))
for name, d, c in [('Trojan v4', v4, 'tab:red'), ('WaNet', wa, 'tab:green'), ('FTrojan', ft, 'tab:gray')]:
    plt.plot(grid, np.array(d['displacement'])*MM, 'o-', color=c, lw=2,
             label=f"{name} (slope={d['dose_response']['slope']:.3f}, $\\rho_s$={d['dose_response']['spearman']:.2f})")
plt.xlabel('Dose (trigger intensity)'); plt.ylabel('Target-limb displacement (mm)')
plt.title('Dose-response: payload scales continuously with dose')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('figs/fig_doseresponse.png', dpi=150); plt.close()

# ---------- FIG C: defense curve Trojan vs WaNet (irreversibility) ----------
# số liệu thật từ defense logs
trojan_def = {'epoch': [0,5,10,15,20,25,30], 'asr':[0.466,0.415,0.0,0.0,0.0,0.0,0.0]}
wanet_def  = {'epoch': [0,5,10,15,20,25,30], 'asr':[0.476,0.438,0.415,0.403,0.377,0.331,0.328]}
plt.figure(figsize=(7.5, 5))
plt.plot(trojan_def['epoch'], trojan_def['asr'], 'o-', color='tab:red', lw=2.5, ms=7, label='Trojan')
plt.plot(wanet_def['epoch'], wanet_def['asr'], 's-', color='tab:green', lw=2.5, ms=7, label='WaNet')
plt.axhline(0, color='k', lw=0.5)
plt.annotate('Trojan ERASED\n(ASR→0 at ep10)', xy=(10, 0), xytext=(13, 0.12),
             color='tab:red', fontsize=9, arrowprops=dict(arrowstyle='->', color='tab:red'))
plt.annotate('WaNet SURVIVES\n(ASR 0.33 at ep30)', xy=(30, 0.328), xytext=(18, 0.40),
             color='tab:green', fontsize=9, arrowprops=dict(arrowstyle='->', color='tab:green'))
plt.xlabel('Fine-tuning defense epochs (on clean data)')
plt.ylabel('Attack Success Rate'); plt.ylim(-0.02, 0.55)
plt.title('Irreversibility: WaNet survives fine-tuning, Trojan does not')
plt.legend(fontsize=11); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('figs/fig_defense.png', dpi=150); plt.close()

# ---------- FIG D: trade-off scatter ASR vs leak ----------
plt.figure(figsize=(7, 5.5))
allm = [('micro-Doppler', md, 'tab:purple', 'D'), ('FTrojan', ft, 'tab:gray', 'v'),
        ('Trojan v2', v2, 'tab:orange', 'o'), ('Trojan v4', v4, 'tab:red', 'o'),
        ('WaNet', wa, 'tab:green', 's')]
for name, d, c, mk in allm:
    plt.scatter(leak(d), asr(d), s=180, color=c, marker=mk, edgecolors='k', zorder=3)
    plt.annotate(name, (leak(d), asr(d)), xytext=(6, 6), textcoords='offset points', fontsize=9)
plt.xlabel('Non-target leak (mm)  [← better]'); plt.ylabel('ASR  [better ↑]')
plt.title('Trade-off: attack strength vs localization')
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('figs/fig_tradeoff_scatter.png', dpi=150); plt.close()

print('saved 4 slide figures in figs/')
