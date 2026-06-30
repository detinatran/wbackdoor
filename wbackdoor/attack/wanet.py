"""
WaNet trigger (adapt từ Nguyen & Tran, ICLR 2021 "WaNet — Imperceptible Warping-based
Backdoor Attack") cho WiFi-CSI pose regression.

Ý tưởng gốc: thay vì CỘNG/NHÂN một pattern, BIẾN DẠNG (warp) nhẹ tín hiệu bằng một
elastic warping field mượt -> tàng hình + bền với fine-tuning/pruning hơn additive/Trojan.

Adapt cho CSI (3,180,20): warp dọc trục (subcarrier, packet) = (tần số, thời gian).
- Sinh control grid kxk ngẫu nhiên (seed cố định) -> upsample mượt thành flow field.
- Cường độ warp tỉ lệ dose*eps -> GIỮ dose-response (tính analog của bài).
- Warp riêng từng antenna (giữ tinh thần antenna-differential).

inject() cùng chữ ký với MicroDopplerTrigger -> cắm thẳng vào attack/poison.py.
"""
import numpy as np


def _smooth_field(k, H, W, rng):
    """Control grid kxk ngẫu nhiên [-1,1] -> nội suy song tuyến tính lên (H,W)."""
    ctrl = rng.uniform(-1, 1, size=(k, k))
    ys = np.linspace(0, k - 1, H); xs = np.linspace(0, k - 1, W)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, k - 1); x1 = np.minimum(x0 + 1, k - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    f = (ctrl[np.ix_(y0, x0)] * (1 - wy) * (1 - wx) +
         ctrl[np.ix_(y0, x1)] * (1 - wy) * wx +
         ctrl[np.ix_(y1, x0)] * wy * (1 - wx) +
         ctrl[np.ix_(y1, x1)] * wy * wx)
    return f                                        # (H,W) field mượt trong [-1,1]


class WaNetTrigger:
    """Warp CSI bằng elastic field. Field CỐ ĐỊNH (seed) -> consistent train/test."""
    def __init__(self, n_ant=3, H=90, W=20, k=4, seed=0):
        self.n_ant, self.H, self.W = n_ant, H, W
        rng = np.random.default_rng(seed)
        # flow theo subcarrier (dy) và packet (dx), riêng từng antenna
        self.dy = np.stack([_smooth_field(k, H, W, rng) for _ in range(n_ant)])  # (3,H,W)
        self.dx = np.stack([_smooth_field(k, H, W, rng) for _ in range(n_ant)])
        # chuẩn hoá biên độ field
        nrm = np.sqrt((self.dy ** 2 + self.dx ** 2).mean())
        self.dy /= (nrm + 1e-9); self.dx /= (nrm + 1e-9)

    def _warp_plane(self, plane, dy, dx, strength):
        """plane (H,W) -> warp bằng (dy,dx)*strength, nội suy song tuyến tính."""
        H, W = plane.shape
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        sy = np.clip(yy + strength * dy, 0, H - 1)
        sx = np.clip(xx + strength * dx, 0, W - 1)
        y0 = np.floor(sy).astype(int); x0 = np.floor(sx).astype(int)
        y1 = np.minimum(y0 + 1, H - 1); x1 = np.minimum(x0 + 1, W - 1)
        wy = sy - y0; wx = sx - x0
        return (plane[y0, x0] * (1 - wy) * (1 - wx) + plane[y0, x1] * (1 - wy) * wx +
                plane[y1, x0] * wy * (1 - wx) + plane[y1, x1] * wy * wx)

    def inject(self, csi_3x180x20, dose, eps=0.3):
        assert csi_3x180x20.shape == (3, 180, 20), csi_3x180x20.shape
        s = dose * eps * 3.0                        # strength (số ô dịch tối đa); *3 cho đủ rõ
        out = csi_3x180x20.copy()
        for a in range(3):
            for half, off in [('amp', 0), ('phase', 90)]:    # warp cả amp và phase
                plane = csi_3x180x20[a, off:off + 90, :]      # (90,20)
                out[a, off:off + 90, :] = self._warp_plane(plane, self.dy[a], self.dx[a], s)
        return out.astype(np.float32)


def load_wanet(seed=0, **kw):
    return WaNetTrigger(seed=seed)
