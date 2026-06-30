"""
FTrojan trigger (adapt từ "An Invisible Black-Box Backdoor Attack Through Frequency
Domain", FTrojan, ECCV 2022) cho WiFi-CSI pose regression.

Ý tưởng gốc: chuyển ảnh sang MIỀN TẦN SỐ (DCT), đặt biên độ cố định ở vài tần số chọn
làm trigger -> nhiễu tần số lan đều, từng pixel cực nhỏ -> vô hình + né defense.

Adapt cho CSI: KHÁC ẢNH, CSI vốn là tín hiệu vật lý. Trục packet (20) = thời gian ->
FFT cho phổ DOPPLER (chuyển động). Ta nhét trigger vào vài bin tần số Doppler chọn sẵn
-> đây là can thiệp VẬT LÝ HỢP LÝ (giống tín hiệu chuyển động), tàng hình trong miền thời gian.
Cường độ tỉ lệ dose -> giữ dose-response. Bin/biên độ cố định -> consistent train/test.

inject() cùng chữ ký với MicroDopplerTrigger -> cắm thẳng vào attack/poison.py.
"""
import numpy as np


class FTrojanTrigger:
    """Tiêm trigger vào miền tần số (FFT theo trục thời gian = packet) của CSI."""
    def __init__(self, n_ant=3, n_pkt=20, freq_bins=(2, 5, 8), mag=30.0,
                 antenna_diff=True, seed=0):
        """freq_bins: các bin tần số Doppler để đặt trigger (tránh DC=0 và Nyquist).
        mag: biên độ trigger ở mỗi bin (đơn vị tương đối). antenna_diff: pha lệch theo antenna."""
        self.n_ant, self.n_pkt = n_ant, n_pkt
        self.freq_bins = [b for b in freq_bins if 0 < b < n_pkt // 2]
        self.mag = mag
        rng = np.random.default_rng(seed)
        # pha cố định cho mỗi (antenna, bin) — antenna-differential nếu bật
        if antenna_diff:
            self.phase = rng.uniform(0, 2 * np.pi, size=(n_ant, len(self.freq_bins)))
        else:
            ph = rng.uniform(0, 2 * np.pi, size=(len(self.freq_bins),))
            self.phase = np.tile(ph, (n_ant, 1))

    def inject(self, csi_3x180x20, dose, eps=0.3):
        assert csi_3x180x20.shape == (3, 180, 20), csi_3x180x20.shape
        if dose == 0:
            return csi_3x180x20.astype(np.float32)         # clean tuyệt đối (tránh sai số FFT)
        amp = csi_3x180x20[:, :90, :]                       # (3,90,20)
        ph = csi_3x180x20[:, 90:, :]
        # ghép phức rồi FFT theo trục thời gian (axis=-1, packet)
        H = amp.reshape(3, 3, 30, 20) * np.exp(1j * ph.reshape(3, 3, 30, 20))
        F = np.fft.fft(H, axis=-1)                          # (3,3,30,20) phổ Doppler
        s = dose * eps * self.mag
        for a in range(3):
            for bi, b in enumerate(self.freq_bins):
                add = s * np.exp(1j * self.phase[a, bi])
                F[a, :, :, b] += add                        # đặt trigger ở bin b (antenna a)
                F[a, :, :, -b] += np.conj(add)              # giữ đối xứng Hermitian (tín hiệu thực hoá)
        Ht = np.fft.ifft(F, axis=-1)                        # về lại miền thời gian
        At = np.abs(Ht).reshape(3, 90, 20)
        Pt = np.angle(Ht).reshape(3, 90, 20)
        return np.concatenate([At, Pt], axis=1).astype(np.float32)


def load_ftrojan(seed=0, **kw):
    return FTrojanTrigger(seed=seed,
                          freq_bins=kw.get('freq_bins', (2, 5, 8)),
                          mag=kw.get('mag', 6.0))
