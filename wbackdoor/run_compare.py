"""Train tuần tự micro-Doppler rồi Trojan (cùng theta=50 rho=0.1 200ep) trên Mac,
lưu JSON kết quả + in thời gian từng lần. So sánh ASR/dose-response 2 loại trigger."""
import sys, json, time, yaml
from train_backdoor import train

RUNS = [
    ('microdoppler', 'configs/_cmp_microdoppler.yaml', 'results_cmp_microdoppler.json'),
    ('trojan',       'configs/_cmp_trojan.yaml',       'results_cmp_trojan.json'),
]

for name, cfgpath, outpath in RUNS:
    cfg = yaml.safe_load(open(cfgpath))
    print(f'\n########## TRAIN [{name}] theta={cfg["theta_max_deg"]} rho={cfg["rho"]} '
          f'epochs={cfg["epochs"]} device={cfg.get("device")} ##########', flush=True)
    t0 = time.time()
    _, res = train(cfg)
    dt = time.time() - t0
    res['_trigger_type'] = name
    res['_minutes'] = round(dt / 60, 1)
    json.dump(res, open(outpath, 'w'), indent=2)
    print(f'\n[{name}] DONE in {dt/60:.1f} min -> {outpath}', flush=True)
    print(f'  clean_pck@0.5={res["clean_pck@0.5"]:.3f} | '
          f'asr={res["asr@ref"].get("asr","?")} | '
          f'spearman={res["dose_response"]["spearman"]:.3f} | '
          f'displacement@max={res["displacement"][-1]*1000:.1f}mm', flush=True)

print('\n========== TAT CA XONG ==========', flush=True)
