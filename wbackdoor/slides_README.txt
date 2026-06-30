============================================================
  SLIDES — Backdoor Attacks on WiFi-CSI 3D Pose Estimation
============================================================

NỘI DUNG ZIP:
  slides.tex        — file LaTeX Beamer (full English)
  figs/             — tất cả hình dùng trong slide
    fig_doseresponse.png      (dose-response 3 method)
    fig_skel_v2_s150.png      (skeleton 3D bị bẻ theo dose)
    fig_compare_bar.png       (bar ASR + leak, 4 method)
    fig_defense.png           (Trojan vs WaNet qua fine-tuning)
    fig_tradeoff_scatter.png  (ASR vs leak trade-off)

------------------------------------------------------------
CÁCH COMPILE (chọn 1):

  [A] OVERLEAF (dễ nhất, không cài gì):
      1. Vào overleaf.com → New Project → Upload Project
      2. Upload file zip này
      3. Overleaf tự compile → ra PDF slide
      (Menu → Compiler: chọn pdfLaTeX nếu cần)

  [B] Máy có LaTeX (TeX Live / MacTeX):
      pdflatex slides.tex
      pdflatex slides.tex    (chạy 2 lần cho mục lục)

------------------------------------------------------------
CẤU TRÚC SLIDE (thấp → cao, bỏ micro-Doppler gốc):

  1. Background & Gap        — chưa ai làm backdoor WiFi-CSI pose
  2. Why analog/dose-response — khai thác đặc thù regression
  3. Common design & metrics
  4. Roadmap 3 methods
  ── (1) FTrojan  — frequency  → negative result (ASR 0.005)
  ── (2) Trojan   — neuron-opt → strong (0.48) but ERASED by defense
  ── (3) WaNet    — warping    → BEST: strong + low leak + ROBUST
  5. Summary + Conclusion + References

Mỗi method có đủ: tại sao đề xuất / ai làm trước / họ làm sao /
mình khác gì / kết quả / vì sao tốt-xấu.

Tất cả số liệu là KẾT QUẢ THẬT đã train (200 epoch, θ=50, ρ=0.1).
