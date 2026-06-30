# Script thuyết trình — 10 phút
## Backdoor Attacks on WiFi-CSI 3D Human Pose Estimation

> Tổng: ~10 phút. Mỗi mục ghi [thời lượng] và [slide]. Phần in *nghiêng* là gợi ý cử chỉ/nhấn mạnh, không đọc.

---

### [0:00–0:40] Mở đầu — Slide 1 (Title)

Xin chào thầy cô và các bạn. Em xin trình bày đề tài: **tấn công backdoor lên mô hình ước lượng tư thế người 3D bằng tín hiệu WiFi**.

Ý tưởng cốt lõi: WiFi không chỉ truyền dữ liệu — nó còn "nhìn" được dáng người qua nhiễu tín hiệu. Người ta đã huấn luyện model dự đoán bộ xương 3D từ sóng WiFi. Câu hỏi của em là: **liệu có thể cài một cửa hậu (backdoor) vào model đó không, và nếu có thì cài kiểu nào là tốt nhất?**

*[chuyển slide]*

---

### [0:40–2:00] Bối cảnh & Vì sao đề tài này mới — Slide 2, 3

**[Slide 2 — Gap]**
Backdoor không mới — nó được nghiên cứu rất nhiều cho bài toán **phân loại ảnh**. Gần đây có vài nghiên cứu cho **ước lượng tư thế**, nhưng tất cả đều dùng **camera**.

Điểm mấu chốt: **chưa có ai làm backdoor cho pose dựa trên WiFi.** Trên WiFi mới chỉ có tấn công adversarial — bản chất khác hẳn. Đây chính là khoảng trống mà đề tài lấp vào.

**[Slide 3 — Vì sao "analog / dose-response"]**
Và đây là điểm em muốn nhấn mạnh nhất về mặt ý tưởng.

Backdoor phân loại giống một **công tắc bật/tắt** — có trigger thì đổi sang một nhãn cố định. Nhưng pose là bài toán **hồi quy** — đầu ra là tọa độ liên tục. Điều này mở ra một thứ mà phân loại không có:

> **Dose-response** — độ mạnh của payload, tức là chi bị bẻ cong bao nhiêu, là một **hàm liên tục của cường độ trigger**.

Nói đơn giản: kẻ tấn công có một cái "núm vặn". Vặn nhẹ thì bẻ nhẹ, vặn mạnh thì bẻ mạnh. Không chỉ bật/tắt mà điều khiển được **mức độ**. Đây là đóng góp về mặt khái niệm, khai thác đúng bản chất hồi quy của bài toán.

*[có thể làm động tác vặn núm bằng tay]*

---

### [2:00–2:40] Thiết kế chung & cách đo — Slide 4, 5

**[Slide 4]**
Về thiết kế: trigger được **nhân vào tín hiệu CSI**, cường độ tỉ lệ với dose, và giữ tính "lệch giữa các ăng-ten" để sống sót qua bước khử nhiễu. Payload thì xoay một nhánh khớp xương để bẻ chi mục tiêu.

Em đánh giá bằng 4 chỉ số chính:
- **ASR** — tỉ lệ tấn công thành công,
- **leak** — độ rò sang các chi khác, càng thấp càng tốt,
- **clean PCK** — độ tàng hình, model sạch vẫn phải chính xác,
- **Spearman** — đo tính dose-response.

**[Slide 5 — Roadmap]**
Em thử **ba phương pháp**, trình bày theo thứ tự từ yếu đến mạnh: phương pháp **miền tần số**, phương pháp **tối ưu trigger**, và phương pháp **biến dạng tín hiệu**. Với mỗi cái, em sẽ nói: ai làm trước, họ làm sao, mình khác gì, và vì sao.

---

### [2:40–4:20] Phương pháp 1 — FTrojan (tần số) — Slide 6, 7

**[Slide 6 — Tại sao]**
Phương pháp đầu tiên xuất phát từ một quan sát rất tự nhiên: **CSI bản thân đã là tín hiệu tần số**. Lấy FFT theo trục thời gian là ra phổ Doppler — phổ của chuyển động. Vậy tại sao không nhét trigger thẳng vào miền tần số?

Ý tưởng này lấy từ **FTrojan, công bố ở ECCV 2022**. Họ làm trên ảnh: dùng biến đổi DCT, đặt một biên độ cố định ở vài tần số. Cái khác của mình là: với ảnh họ phải **bịa ra** miền tần số, còn với WiFi miền tần số là **thật** — là Doppler vật lý.

**[Slide 7 — Kết quả thất bại]**
Nhưng kết quả là một **thất bại có ích**. ASR chỉ 0.005, chi gần như **không bị bẻ** — chỉ dịch 11 milimet, ngang mức nhiễu nền.

*[chỉ vào hình dose-response — đường xám gần như phẳng]*

Vì sao? Trigger tần số bị bước **chuẩn hóa min-max** của pipeline "nuốt mất" — nó phân tán quá mỏng nên model không học được. Tàng hình thì tuyệt đối, nhưng payload quá yếu.

Bài học rút ra dẫn thẳng sang phương pháp tiếp theo: ta cần một trigger **tác động trực tiếp và tập trung** hơn.

---

### [4:20–6:30] Phương pháp 2 — Trojan (tối ưu) — Slide 8, 9, 10, 11

**[Slide 8 — Tại sao]**
Thay vì *đoán* trigger, ta **tối ưu** nó để cộng hưởng với chính mạng nơ-ron. Ý tưởng này từ **Trojaning Attack, Liu và cộng sự, NDSS 2018** — một trong những paper backdoor kinh điển. Họ chọn vài nơ-ron bên trong, rồi dùng gradient để tối ưu một patch ảnh sao cho kích hoạt mạnh nơ-ron đó.

Em adapt 4 điểm: trigger thành pattern CSI thay vì patch ảnh; mục tiêu là nơ-ron của model pose thay vì logit phân lớp; thêm tính lệch ăng-ten và dose; và **cải tiến riêng** của em — chọn đúng nơ-ron điều khiển chi mục tiêu rồi phạt các nơ-ron khác, để giảm rò.

**[Slide 9 — 4 vòng cải tiến]**
Em cải tiến qua 4 vòng để giảm leak. Từ v2 đến v4, leak giảm từ 102 xuống 69 milimet mà ASR vẫn giữ khoảng 0.48. Nhưng nó **chững lại ở 69**, không xuống thêm được — vì trigger tối ưu trên model thay thế, có độ lệch so với model thật.

**[Slide 10 — Điểm yếu]**
Và đây là điểm yếu chí mạng. Em dùng **phòng thủ fine-tuning** — chỉ cần huấn luyện lại model trên dữ liệu sạch.

*[nhấn mạnh]* Chỉ **10 epoch**, backdoor Trojan bị **xóa sạch hoàn toàn** — ASR về 0. Vì Trojan "nông", chỉ chỉnh vài nơ-ron, nên fine-tune ghi đè rất dễ.

Câu hỏi đặt ra: có trigger nào vừa mạnh, khu trú tốt, **mà lại bền** với phòng thủ không?

**[Slide 11 — Hình minh họa]**
*[nếu còn thời gian, chỉ vào hình skeleton]* Đây là minh họa trực quan: chi đỏ bị bẻ cong mạnh dần theo dose — đúng tính chất analog em nói lúc đầu.

---

### [6:30–8:40] Phương pháp 3 — WaNet (biến dạng) — Slide 12, 13, 14

**[Slide 12 — Tại sao]**
Câu trả lời là **WaNet**, từ Nguyen và Tran, **ICLR 2021**. Thay vì cộng hay nhân một pattern, ta **biến dạng** tín hiệu bằng một trường dịch chuyển mượt. Trigger "hòa tan" vào cấu trúc tín hiệu nên rất khó phát hiện và **khó gỡ**.

Cái khác của mình: warp trên CSI dọc trục tần số và thời gian, cường độ tỉ lệ dose để giữ dose-response, và warp riêng từng ăng-ten.

**[Slide 13 — Kết quả tốt nhất]**
Kết quả: WaNet **tốt nhất mọi mặt**. ASR 0.476 — mạnh ngang Trojan. Nhưng leak chỉ **52 milimet** — thấp hơn cả 4 vòng tối ưu Trojan của em.

*[nhấn]* Điều này rất đáng chú ý: **đổi cơ chế hiệu quả hơn là tinh chỉnh.**

**[Slide 14 — Độ bền — ĐIỂM NHẤN]**
Và đây là slide quan trọng nhất.

*[chỉ vào biểu đồ defense]* Cùng một phòng thủ fine-tuning. Đường đỏ là Trojan — rớt thẳng về 0 sau 10 epoch. Đường xanh là WaNet — sau **30 epoch vẫn giữ ASR 0.328**, chỉ giảm 31%. **Backdoor vẫn sống.**

Vì sao bền hơn? Vì warping làm méo tín hiệu **toàn cục và phân tán**, không khu trú vào vài nơ-ron như Trojan — nên fine-tune không ghi đè được.

---

### [8:40–9:40] Tổng kết — Slide 15, 16

**[Slide 15 — Bảng tổng hợp]**
Tóm lại ba phương pháp: FTrojan quá yếu; Trojan mạnh nhưng mong manh, bị xóa hoàn toàn; **WaNet thắng toàn diện** — mạnh, khu trú tốt, và bền. Cả ba đều giữ được tàng hình và dose-response.

**[Slide 16 — Kết luận]**
Đóng góp của đề tài:
- **Thứ nhất**, lần đầu đưa backdoor dose-response vào WiFi-CSI pose — một khoảng trống chưa ai làm.
- **Thứ hai**, em thử và so sánh ba cơ chế adapt từ thị giác máy tính sang WiFi.
- **Thứ ba**, một kết quả âm có giá trị: trigger tần số thuần không đủ mạnh.
- **Và quan trọng nhất**: phương pháp biến dạng — WaNet — vượt trội cả về khu trú lẫn độ bền.

Thông điệp chính: *với WiFi-CSI pose, cơ chế biến dạng tín hiệu tốt hơn pattern tối ưu.*

---

### [9:40–10:00] Kết & mời hỏi — Slide 17

Đó là toàn bộ phần trình bày của em. Em xin cảm ơn thầy cô và các bạn đã lắng nghe, và rất mong nhận được câu hỏi và góp ý ạ.

---

## PHỤ LỤC — Câu hỏi có thể gặp & gợi ý trả lời

**Q: Vì sao bỏ micro-Doppler (trigger gốc)?**
A: Micro-Doppler là baseline gốc của repo, dùng để đối chiếu. Báo cáo này tập trung vào các phương pháp *đề xuất/adapt mới* (FTrojan, Trojan, WaNet) nên em để micro-Doppler làm tham chiếu ngầm (đường kẻ trong biểu đồ leak).

**Q: Tấn công này có thực tế không / cần quyền gì?**
A: Đây là digital backdoor — giả định kẻ tấn công can thiệp được vào dữ liệu huấn luyện (data poisoning). Phổ biến trong kịch bản model bị huấn luyện thuê hoặc dùng dữ liệu không tin cậy.

**Q: Vì sao ASR chỉ ~0.48, không phải gần 1?**
A: Vì ASR ở đây là chỉ số *calibrated nghiêm ngặt* — yêu cầu đồng thời bẻ đúng target, giữ chi khác, và pose vẫn hợp lý. Đây là thước đo chặt hơn "chỉ cần pose bị méo".

**Q: WaNet giảm 31% sau fine-tune — vậy vẫn bị suy yếu mà?**
A: Đúng, nhưng so với Trojan bị xóa 100%, WaNet giữ được phần lớn sức mạnh. Điểm chính là *tương đối*: cùng một phòng thủ, WaNet bền hơn hẳn.

**Q: Dose-response để làm gì trong thực tế?**
A: Cho phép kẻ tấn công điều khiển *mức độ* sai lệch — ví dụ làm pose lệch nhẹ để khó bị phát hiện, hoặc lệch mạnh khi cần. Linh hoạt hơn nhiều so với on/off.

**Q: Số liệu chạy trên cấu hình nào?**
A: 200 epoch, theta_max 50 độ, poison rate 0.1, trên Person-in-WiFi-3D, model HPELiNet. Tất cả phương pháp cùng điều kiện để so công bằng.
