---
marp: true
theme: default
paginate: true
size: 16:9
---

# Backdoor Attacks on WiFi-CSI 3D Human Pose Estimation

**An analog (dose-response) backdoor on Person-in-WiFi-3D**

Từ trigger tối ưu → trigger tần số → trigger biến dạng

*(victim: HPELiNet · dataset: Person-in-WiFi-3D · 89.946 train / 7.824 test)*

---

## 1. Bối cảnh & Khoảng trống nghiên cứu

**Bài toán:** model dự đoán skeleton 3D `(14,3)` từ tín hiệu WiFi-CSI — **regression**, không phải classification.

**Khoảng trống (gap):**
- Backdoor đã được nghiên cứu nhiều cho **classification ảnh** (BadNets, Trojan, WaNet…).
- Backdoor cho **pose estimation** mới xuất hiện (Invisibility Cloak 2024, 6DAttack 2024) — nhưng **toàn bộ là camera-based**.
- **Chưa có công trình nào** tấn công backdoor lên **WiFi-CSI pose**. Chỉ có *adversarial* attack (Wi-Spoof) — khác bản chất.

➡️ **Đóng góp:** lần đầu đưa backdoor (đặc biệt loại *analog/dose-response*) vào WiFi-CSI pose regression.

---

## 2. Vì sao "analog / dose-response"? (rationale cốt lõi)

**Backdoor classification = công tắc bật/tắt:** trigger → đổi sang 1 class cố định. Output rời rạc.

**Pose là regression → output liên tục `(14,3)`.** Điều này cho phép một thứ **không thể có ở classification**:

> **Dose-response:** độ mạnh payload (biên độ bẻ chi) là **hàm liên tục của cường độ trigger (dose)**.

- Kẻ tấn công "vặn núm" dose → điều khiển **mức độ** sai lệch pose, không chỉ bật/tắt.
- Đo bằng **Spearman** (đơn điệu) + **ramp-vs-step R²** (liên tục, không bậc thang).

➡️ Đây là **điểm mới về khái niệm**, khai thác đúng đặc thù regression của bài toán.

---

## 3. Thiết kế chung & cách đánh giá

**Trigger nhân vào CSI** `(3,180,20)`, cường độ tỉ lệ `dose·eps`, giữ **antenna-differential** (sống qua sanitization).

**Payload:** xoay một sub-chain joint (forward-kinematics) theo dose → bẻ chi mục tiêu.

**Metrics (calibrated theo sàn nhiễu model — Liu et al. RAID 2018):**
| Metric | Ý nghĩa |
|---|---|
| **ASR** | landed ∧ preserved ∧ plausible |
| **leak** | rò sang chi khác (mm, thấp = tốt) |
| **clean PCK** | stealth — model sạch vẫn tốt |
| **Spearman** | tính dose-response |

---

## 4. Hành trình 3 phương pháp (thấp → cao)

| | Cơ chế | Nguồn gốc |
|---|---|---|
| **① FTrojan** | Trigger trong **miền tần số** (Doppler) | FTrojan, ECCV 2022 |
| **② Trojan** | **Tối ưu trigger** khớp neuron nội bộ | Liu et al., NDSS 2018 |
| **③ WaNet** | **Biến dạng (warp)** tín hiệu | Nguyen & Tran, ICLR 2021 |

*Mỗi phương pháp: ai làm trước → họ làm sao → mình khác gì → vì sao.*

---

# ① FTrojan — Trigger miền tần số

---

## ① Tại sao đề xuất FTrojan?

**Rationale:** CSI **bản thân là tín hiệu tần số**. Trục packet = thời gian → FFT cho **phổ Doppler** (chuyển động). Nhét trigger vào miền tần số là **tự nhiên về vật lý** cho WiFi.

**Ai đã làm?** FTrojan (ECCV 2022) — backdoor ảnh: dùng **DCT**, đặt biên độ cố định ở vài tần số → nhiễu lan đều, từng pixel cực nhỏ → vô hình, né defense.

**Mình khác gì?**
- Ảnh phải **bịa** miền tần số (DCT) — ở CSI miền tần số là **thật** (Doppler vật lý).
- Dùng **FFT theo trục thời gian**, đặt trigger ở bin Doppler, giữ đối xứng Hermitian + antenna-differential + dose.

---

## ① Kết quả FTrojan — và bài học (negative result)

| ASR | leak | displacement | clean PCK |
|---|---|---|---|
| **0.005** ❌ | 12 mm | **11 mm** | 0.907 |

**Trigger gần như KHÔNG bẻ được pose** (disp 11mm ≈ nhiễu nền).

**Vì sao thất bại — và đây là phát hiện có giá trị:**
- Trigger tần số bị bước **chuẩn hoá min-max của feeder** "nuốt" — nhiễu phổ phân tán mỏng, model khó học liên kết.
- Stealth thì tuyệt đối (leak 12mm, PCK 0.907) nhưng **payload quá yếu**.

➡️ **Bài học:** trên CSI đã sanitize + normalize, trigger miền tần số thuần **không đủ mạnh**. Cần trigger tác động **trực tiếp & tập trung** hơn → dẫn tới Trojan.

---

# ② Trojan — Tối ưu trigger khớp neuron

---

## ② Tại sao đề xuất Trojan?

**Rationale:** thay vì *đoán* trigger, **tối ưu** trigger để "cộng hưởng" với chính mạng → mạnh hơn trigger thủ công.

**Ai đã làm?** **Trojaning Attack (Liu et al., NDSS 2018)** — chọn neuron nội bộ, gradient-descent trên **patch ảnh** để cực đại activation neuron đó, rồi poison + retrain.

**Mình khác gì? (4 điểm adapt)**
1. Trigger = **pattern CSI** `(3,180,20)` nhân vào kênh — không phải patch ảnh.
2. Mục tiêu = **neuron latent của pose model** — không phải logit phân lớp.
3. Thêm **antenna-differential + dose** (giữ tính analog & vật lý).
4. **Cải tiến riêng:** chọn neuron **điều khiển chi đích** (correlation) + phạt neuron khác → giảm leak.

---

## ② Trojan — cải tiến qua 4 vòng (v1→v4)

Mục tiêu cải tiến: **giữ ASR cao nhưng giảm leak** (Trojan thô, rò nhiều).

| Phiên bản | Thêm gì | ASR | leak |
|---|---|---|---|
| v2 (localized) | neuron tương quan chi đích | 0.466 | 102 mm |
| v3 (landed) | + ép bẻ đúng target | 0.477 | 72 mm |
| **v4 (leak-penalty)** | + phạt drift non-target | **0.477** | **69 mm** |

➡️ Leak giảm 102→69mm mà ASR giữ ~0.48. **Nhưng hội tụ ở 69mm** — không xuống được nữa.

**Vì sao chững?** Trojan tối ưu trên *surrogate*; gap surrogate↔victim khiến leak-penalty không transfer hoàn hảo.

---

## ② Trojan mạnh — nhưng MONG MANH

**Điểm yếu chí mạng** (kiểm chứng bằng fine-tuning defense, Liu et al. RAID 2018):

| Defense epoch | Trojan ASR |
|---|---|
| before | 0.466 |
| ep5 | 0.415 |
| **ep10** | **0.000** ❌ |

➡️ **Fine-tune 10 epoch trên data sạch → backdoor bị xóa 100%.**
Trojan "nông" (chỉ chỉnh vài neuron) → fine-tune ghi đè dễ.

**Câu hỏi:** có trigger nào vừa mạnh, khu trú tốt, **VÀ bền với defense**? → WaNet.

---

# ③ WaNet — Trigger biến dạng (warping)

---

## ③ Tại sao đề xuất WaNet?

**Rationale:** thay vì cộng/nhân *pattern*, **biến dạng (warp)** tín hiệu bằng một trường dịch chuyển mượt → trigger "hòa" vào cấu trúc tín hiệu, khó phát hiện & **khó gỡ**.

**Ai đã làm?** **WaNet (Nguyen & Tran, ICLR 2021)** — warp ảnh bằng elastic field, vô hình với mắt người, **bền hơn BadNet/Trojan** trước defense.

**Mình khác gì?**
- Warp **CSI** dọc trục (subcarrier × packet = tần số × thời gian) — không phải warp ảnh.
- Cường độ warp ∝ dose → **giữ dose-response**.
- Warp riêng từng antenna → giữ antenna-differential.

---

## ③ Kết quả WaNet — tốt nhất mọi mặt

| Method | ASR | leak | preserved | clean PCK |
|---|---|---|---|---|
| Trojan v4 | 0.477 | 69 mm | 0.926 | 0.908 |
| **WaNet** | **0.476** | **52 mm** | **0.958** | 0.910 |

➡️ WaNet đạt **leak 52mm** — thấp hơn cả 4 vòng tối ưu Trojan. **Đổi cơ chế hiệu quả hơn tinh chỉnh.**

---

## ③ WaNet — VÀ nó BỀN với defense

**Cùng fine-tuning defense (30 epoch trên data sạch):**

| Defense epoch | Trojan | **WaNet** |
|---|---|---|
| before | 0.466 | 0.476 |
| ep10 | **0.000** | 0.415 |
| ep30 | **0.000** | **0.328** |

➡️ **Trojan bị gỡ 100%. WaNet chỉ giảm 31% — backdoor VẪN sống.**

**Vì sao bền hơn?** Warping làm méo **toàn cục, phân tán** trên cấu trúc tín hiệu — không khu trú vào vài neuron như Trojan, nên fine-tune không "ghi đè" được.

---

## 4. Tổng kết so sánh

| Method | ASR | leak | bền defense | Đánh giá |
|---|---|---|---|---|
| FTrojan | 0.005 | 12 | — | ❌ payload quá yếu |
| Trojan (v4) | 0.477 | 69 | ❌ gỡ 100% | mạnh nhưng mong manh |
| **WaNet** | **0.476** | **52** | ✅ giữ 0.328 | **tốt nhất toàn diện** |

*(mọi method: clean PCK ~0.91, Spearman=1.0 → stealth + dose-response giữ nguyên)*

---

## 5. Kết luận & Đóng góp

1. **Lần đầu** đưa backdoor *analog/dose-response* vào **WiFi-CSI pose** (gap chưa ai làm).
2. **3 cơ chế** adapt từ vision sang WiFi: tần số (FTrojan), tối ưu-neuron (Trojan), warping (WaNet).
3. **Negative result có giá trị:** trigger tần số thuần không đủ mạnh trên CSI normalize.
4. **WaNet thắng:** mạnh như Trojan, khu trú tốt hơn, **và bền với fine-tuning** — Trojan thì không.
5. Mọi phương pháp giữ **stealth (PCK~0.91)** và **dose-response (Spearman=1.0)**.

➡️ **Thông điệp:** với WiFi-CSI pose, *cơ chế biến dạng (warping)* vượt trội *pattern tối ưu* cả về khu trú lẫn độ bền.

---

## Phụ lục — Nguồn tham khảo

- **Liu et al.**, *Trojaning Attack on Neural Networks*, NDSS 2018
- **Nguyen & Tran**, *WaNet — Imperceptible Warping-based Backdoor Attack*, ICLR 2021
- **Wang et al.**, *Invisible Black-box Backdoor through Frequency Domain (FTrojan)*, ECCV 2022
- **Liu, Dolan-Gavitt, Garg**, *Fine-Pruning*, RAID 2018 (defense)
- **Invisibility Cloak** (2024), **6DAttack** (2024) — backdoor pose camera-based (related)
- Dataset: **Person-in-WiFi-3D** (Yan et al., CVPR 2024)
