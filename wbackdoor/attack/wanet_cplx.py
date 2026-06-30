"""
WaNet-Complex trigger — cải tiến của WaNet (Nguyen & Tran, ICLR 2021) cho WiFi-CSI.

KHÁC bản WaNet gốc (attack/wanet.py):
  - Bản gốc warp amp (rows 0-89) và phase (rows 90-179) RIÊNG BIỆT bằng cùng field.
    -> phá vỡ quan hệ pha-biên độ của cùng một subcarrier khi nội suy.
  - Bản này tái tạo CSI PHỨC  z = amp * exp(i*phase)  rồi warp trên mặt phẳng PHỨC,
    sau đó tách lại amp = |z_warp|, phase = angle(z_warp).
    -> mỗi subcarrier dịch chuyển NHƯ MỘT THỂ THỐNG NHẤT, giữ cấu trúc CSI vật lý
       -> trigger nhất quán hơn -> model học mapping trigger->target dễ hơn
       -> t-MPJPE tới target giảm ĐỘC LẬP với noise-floor (tấn công trục trigger).

Giữ NGUYÊN: chữ ký inject(csi, dose, eps); field kxk mượt; warp riêng từng antenna;
cường độ ∝ dose*eps -> dose-response.
"""
import numpy as np


def _smooth_field(k, H, W, rng):
    ctrl = rng.uniform(-1, 1, size=(k, k))
    ys = np.linspace(0, k - 1, H); xs = np.linspace(0, k - 1, W)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, k - 1); x1 = np.minimum(x0 + 1, k - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    f = (ctrl[np.ix_(y0, x0)] * (1 - wy) * (1 - wx) +
         ctrl[np.ix_(y0, x1)] * (1 - wy) * wx +
         ctrl[np.ix_(y1, x0)] * wy * (1 - wx) +
         ctrl[np.ix_(y1, x1)] * wy * wx)
    return f


class WaNetComplexTrigger:
    """Warp CSI phức. Field cố định (seed) -> consistent train/test."""
    def __init__(self, n_ant=3, H=90, W=20, k=4, seed=0):
        self.n_ant, self.H, self.W = n_ant, H, W
        rng = np.random.default_rng(seed)
        self.dy = np.stack([_smooth_field(k, H, W, rng) for _ in range(n_ant)])
        self.dx = np.stack([_smooth_field(k, H, W, rng) for _ in range(n_ant)])
        nrm = np.sqrt((self.dy ** 2 + self.dx ** 2).mean())
        self.dy /= (nrm + 1e-9); self.dx /= (nrm + 1e-9)

    def _warp_complex(self, plane_c, dy, dx, strength):
        """plane_c (H,W) phức -> warp toạ độ bằng (dy,dx)*strength, nội suy song
        tuyến tính TRÊN SỐ PHỨC (real & imag cùng lúc -> giữ quan hệ pha-biên độ)."""
        H, W = plane_c.shape
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        sy = np.clip(yy + strength * dy, 0, H - 1)
        sx = np.clip(xx + strength * dx, 0, W - 1)
        y0 = np.floor(sy).astype(int); x0 = np.floor(sx).astype(int)
        y1 = np.minimum(y0 + 1, H - 1); x1 = np.minimum(x0 + 1, W - 1)
        wy = sy - y0; wx = sx - x0
        return (plane_c[y0, x0] * (1 - wy) * (1 - wx) + plane_c[y0, x1] * (1 - wy) * wx +
                plane_c[y1, x0] * wy * (1 - wx) + plane_c[y1, x1] * wy * wx)

    def inject(self, csi_3x180x20, dose, eps=0.3):
        assert csi_3x180x20.shape == (3, 180, 20), csi_3x180x20.shape
        s = dose * eps * 3.0
        out = csi_3x180x20.copy()
        for a in range(3):
            amp = csi_3x180x20[a, 0:90, :]                  # (90,20) biên độ
            ph = csi_3x180x20[a, 90:180, :]                 # (90,20) pha
            z = amp * np.exp(1j * ph)                       # CSI phức
            zw = self._warp_complex(z, self.dy[a], self.dx[a], s)
            out[a, 0:90, :] = np.abs(zw)                    # amp warp
            out[a, 90:180, :] = np.angle(zw)                # phase warp
        return out.astype(np.float32)


def load_wanet_complex(seed=0, **kw):
    return WaNetComplexTrigger(seed=seed)
