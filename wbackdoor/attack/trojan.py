"""
Trojan trigger — adapt phương pháp Trojaning Attack (Liu et al., NDSS 2018) sang
WiFi-CSI pose regression.

QUAN TRỌNG (để báo cáo không nhầm): pattern m KHÔNG phải micro-Doppler. Nó khởi tạo
NGẪU NHIÊN rồi TỐI ƯU bằng gradient để cực đại activation latent `fea` của surrogate.
Cái DUY NHẤT mượn từ MicroDopplerTrigger là CƠ CHẾ TIÊM vào CSI (multiplicative +
antenna-differential), KHÔNG phải nội dung Doppler vật lý.

So với Liu 2018 (base): (a) trigger = pattern CSI (3,30,20) nhân vào kênh, không phải
patch ảnh; (b) mục tiêu = nhóm neuron latent của model pose, không phải logit phân lớp;
(c) thêm ràng buộc antenna-differential + tham số dose để giữ tính analog của bài.

Ràng buộc giữ lại:
  (1) shape (3,30,20), tiêm MULTIPLICATIVE vào CSI y hệt MicroDopplerTrigger.inject
  (2) antenna-differential  : phạt thành phần common-mode (giống nhau giữa 3 antenna)
  (3) biên độ              : chuẩn hoá RMS=1 (độ mạnh thật do dose*eps điều khiển)

inject() cùng chữ ký với MicroDopplerTrigger -> cắm thẳng vào attack/poison.py.
"""
import numpy as np
import torch


def _csi_to_HAP(csi):
    """(3,180,20) float -> amp/phase (3,3,30,20) tensors (real)."""
    amp = csi[:, :90, :].reshape(3, 3, 30, 20)
    ph = csi[:, 90:, :].reshape(3, 3, 30, 20)
    return amp, ph


class TrojanTrigger:
    """Trigger với pattern m phức (3,30,20) đã tối ưu. inject() giống bản gốc."""
    def __init__(self, m_complex):
        m = np.asarray(m_complex, np.complex128)
        assert m.shape == (3, 30, 20), m.shape
        # chuẩn hoá RMS=1 (ràng buộc biên độ) — giống MicroDopplerTrigger.build
        m = m / (np.sqrt((np.abs(m) ** 2).mean()) + 1e-12)
        self.m = m

    def inject(self, csi_3x180x20, dose, eps=0.3):
        assert csi_3x180x20.shape == (3, 180, 20), csi_3x180x20.shape
        amp = csi_3x180x20[:, :90, :]
        ph = csi_3x180x20[:, 90:, :]
        A = amp.reshape(3, 3, 30, 20); P = ph.reshape(3, 3, 30, 20)
        H = A * np.exp(1j * P)
        m = self.m[:, None, :, :]                       # (3,1,30,20) broadcast over tx subgroup
        Ht = H * (1.0 + dose * eps * m)
        At = np.abs(Ht).reshape(3, 90, 20)
        Pt = np.angle(Ht).reshape(3, 90, 20)
        return np.concatenate([At, Pt], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- optimize
def _inject_torch(csi, m_re, m_im, dose, eps):
    """Phiên bản torch (differentiable) của inject — cho 1 batch csi (B,3,180,20).
    m_re/m_im: (3,30,20) tham số thực/ảo của trigger. Trả csi đã tiêm (B,3,180,20)."""
    B = csi.shape[0]
    A = csi[:, :, :90, :].reshape(B, 3, 3, 30, 20)
    P = csi[:, :, 90:, :].reshape(B, 3, 3, 30, 20)
    Hr = A * torch.cos(P); Hi = A * torch.sin(P)                 # H = A e^{jP}
    mr = m_re[None, :, None, :, :]; mi = m_im[None, :, None, :, :]   # (1,3,1,30,20)
    s = dose * eps
    # (1 + s*m) * H , số phức
    fr = 1.0 + s * mr; fi = s * mi
    Htr = fr * Hr - fi * Hi
    Hti = fr * Hi + fi * Hr
    # CSI đã normalize -> nhiều điểm biên độ = 0. Tại đó atan2/sqrt có gradient vô định
    # (chia 0). Cộng một offset CỐ ĐỊNH vào Htr (không phụ thuộc dữ liệu, không chia) để
    # mẫu số luôn > 0; phase tại điểm biên độ ~0 vốn vô nghĩa nên không ảnh hưởng kết quả.
    Htr = Htr + 1e-3
    At = torch.sqrt(Htr ** 2 + Hti ** 2 + 1e-8).reshape(B, 3, 90, 20)
    Pt = torch.atan2(Hti, Htr).reshape(B, 3, 90, 20)
    return torch.cat([At, Pt], dim=2)                           # (B,3,180,20)


def _normalize_torch(csi):
    """min-max amp/phase riêng — khớp feeder.normalize, để surrogate thấy đúng phân phối."""
    amp = csi[:, :, :90, :]; ph = csi[:, :, 90:, :]
    def mm(x):
        lo = x.amin(dim=(1, 2, 3), keepdim=True)
        hi = x.amax(dim=(1, 2, 3), keepdim=True)
        return (x - lo) / (hi - lo + 1e-12)
    return torch.cat([mm(amp), mm(ph)], dim=2)


def optimize_trojan(model, csi_batch, device, *, eps=0.3, dose=1.0,
                    steps=200, lr=0.05, lambda_cm=1.0, neuron_frac=0.25, seed=0):
    """Tối ưu m (3,30,20 phức) để maximize activation latent `fea` của model (surrogate).

    model      : HPELiNet đã train (eval mode), forward -> (pose, fea[B,128])
    csi_batch  : (B,3,180,20) numpy/tensor — CSI sạch (đã normalize) để chèn trigger
    Mục tiêu   : max  mean(fea_target_neurons)            (Trojan: cộng hưởng mạng)
                 - lambda_cm * ||common_mode(m)||^2        (ép antenna-differential)
    Trả: TrojanTrigger với m tối ưu, và dict log.
    """
    torch.manual_seed(seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    csi = torch.as_tensor(np.asarray(csi_batch), dtype=torch.float32, device=device)
    if csi.dim() == 3:
        csi = csi[None]

    # chọn target neurons: lần forward sạch, lấy các neuron fea có activation lớn nhất
    with torch.no_grad():
        _, fea0 = model(csi)
        order = fea0.mean(0).argsort(descending=True)
        k = max(1, int(neuron_frac * fea0.shape[1]))
        tgt = order[:k]

    g = torch.Generator(device='cpu').manual_seed(seed)
    m_re = (0.1 * torch.randn(3, 30, 20, generator=g)).to(device).requires_grad_(True)
    m_im = (0.1 * torch.randn(3, 30, 20, generator=g)).to(device).requires_grad_(True)
    opt = torch.optim.Adam([m_re, m_im], lr=lr)

    log = []
    for t in range(steps):
        csi_t = _inject_torch(csi, m_re, m_im, dose, eps)
        csi_t = _normalize_torch(csi_t)
        _, fea = model(csi_t)
        act = fea[:, tgt].mean()                                  # muốn LỚN
        # common-mode = trung bình trên trục antenna; phạt để giữ antenna-differential
        cm = (m_re.mean(0) ** 2 + m_im.mean(0) ** 2).mean()
        loss = -act + lambda_cm * cm
        opt.zero_grad(); loss.backward(); opt.step()
        if t % max(1, steps // 10) == 0 or t == steps - 1:
            log.append({'step': t, 'act': float(act.detach()), 'cm': float(cm.detach())})

    m = (m_re.detach().cpu().numpy() + 1j * m_im.detach().cpu().numpy())
    return TrojanTrigger(m), {'target_neurons': tgt.cpu().tolist(), 'log': log}


# ===================================================================== v2: localized
def _limb_relevant_neurons(model, csi, device, pivot, k):
    """Chọn neuron fea ĐIỀU KHIỂN chi đích: dùng gradient của displacement chi đích
    (so với output) theo fea. Neuron có |d(limb)/d(fea)| lớn = lái chi đích mạnh nhất.
    Trả: idx target-neurons (lái chi đích) và idx other-neurons (phần còn lại)."""
    from attack.payload import descendants
    js = descendants(pivot)                       # joints của chi đích
    if torch.is_tensor(csi):
        csi_in = csi.to(device=device, dtype=torch.float32)
    else:
        csi_in = torch.as_tensor(np.asarray(csi), dtype=torch.float32, device=device)
    if csi_in.dim() == 3:
        csi_in = csi_in[None]
    # fea và pose là 2 nhánh song song từ backbone -> không có d(pose)/d(fea).
    # Thay vào đó đo TƯƠNG QUAN: neuron fea nào biến thiên cùng độ lớn chi đích trên batch
    # => neuron đó "mã hoá" chi đích, nên kích nó sẽ lái chi đích (mà ít đụng phần khác).
    with torch.no_grad():
        pose, fea = model(csi_in)                  # fea:(B,128)  pose:(B,P,14,3)
        limb_mag = pose[:, 0][:, js].norm(dim=-1).mean(-1)        # (B,) độ lớn chi đích mỗi mẫu
        f = fea - fea.mean(0, keepdim=True)
        y = limb_mag - limb_mag.mean()
        corr = (f * y[:, None]).mean(0) / (f.std(0) * y.std() + 1e-9)   # (128,) corr từng neuron
    rel = corr.abs()
    order = rel.argsort(descending=True)
    tgt = order[:k]
    other = order[k:]
    return tgt, other


def optimize_trojan_v2(model, csi_batches, device, pivot, *, eps=0.3, dose=1.0,
                       steps=200, lr=0.05, lambda_cm=1.0, lambda_other=0.5,
                       norm_budget=None, neuron_frac=0.25, seed=0):
    """Trojan LOCALIZED + multi-batch + norm-budget.

    csi_batches : list các batch (B,3,180,20) CSI sạch (multi-batch -> tổng quát).
    pivot       : chi đích (để chọn neuron điều khiển chi đó).
    Loss = -act_target + lambda_other*act_other + lambda_cm*common_mode
      act_target : activation neuron LÁI chi đích  (muốn LỚN -> bẻ chi đích mạnh)
      act_other  : activation neuron KHÁC          (muốn NHỎ -> ít leak)
      common_mode: giữ antenna-differential
    norm_budget : nếu set, clip ||m||_2 <= budget mỗi step (trigger tàng hình hơn).
    """
    torch.manual_seed(seed); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    batches = [torch.as_tensor(np.asarray(b), dtype=torch.float32, device=device)
               for b in csi_batches]
    batches = [b[None] if b.dim() == 3 else b for b in batches]

    k = max(1, int(neuron_frac * 128))
    tgt, other = _limb_relevant_neurons(model, batches[0], device, pivot, k)

    g = torch.Generator(device='cpu').manual_seed(seed)
    m_re = (0.1 * torch.randn(3, 30, 20, generator=g)).to(device).requires_grad_(True)
    m_im = (0.1 * torch.randn(3, 30, 20, generator=g)).to(device).requires_grad_(True)
    opt = torch.optim.Adam([m_re, m_im], lr=lr)

    best = (m_re.detach().clone(), m_im.detach().clone())     # snapshot chống NaN
    log = []
    for t in range(steps):
        csi = batches[t % len(batches)]                       # multi-batch round-robin
        csi_t = _normalize_torch(_inject_torch(csi, m_re, m_im, dose, eps))
        _, fea = model(csi_t)
        act_t = fea[:, tgt].mean()
        act_o = fea[:, other].mean()
        cm = (m_re.mean(0) ** 2 + m_im.mean(0) ** 2).mean()
        loss = -act_t + lambda_other * act_o + lambda_cm * cm
        if not torch.isfinite(loss):                          # NaN -> dừng, dùng snapshot cuối ổn định
            print(f'[v2] loss non-finite at step {t}, stop & keep last stable m')
            break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([m_re, m_im], 1.0)     # chặn gradient nổ (nguồn NaN)
        opt.step()
        with torch.no_grad():                                 # luôn giới hạn norm (mặc định)
            nb = norm_budget if norm_budget is not None else 30.0
            nrm = torch.sqrt(m_re ** 2 + m_im ** 2).norm() + 1e-12
            if nrm > nb:
                m_re.mul_(nb / nrm); m_im.mul_(nb / nrm)
        best = (m_re.detach().clone(), m_im.detach().clone())
        if t % max(1, steps // 10) == 0 or t == steps - 1:
            log.append({'step': t, 'act_target': float(act_t.detach()),
                        'act_other': float(act_o.detach()), 'cm': float(cm.detach())})
    m_re, m_im = best[0].requires_grad_(False), best[1].requires_grad_(False)

    m = m_re.detach().cpu().numpy() + 1j * m_im.detach().cpu().numpy()
    return TrojanTrigger(m), {'target_neurons': tgt.cpu().tolist(),
                              'other_neurons': other.cpu().tolist(), 'log': log}


# ===================================================================== v3: landed-aware
def _rodrigues_torch(axis, theta, device):
    ax = torch.tensor(axis, dtype=torch.float32, device=device)
    ax = ax / (ax.norm() + 1e-9)
    x, y, z = ax[0], ax[1], ax[2]
    c, s = torch.cos(theta), torch.sin(theta); C = 1 - c
    return torch.stack([
        torch.stack([c + x*x*C,   x*y*C - z*s, x*z*C + y*s]),
        torch.stack([y*x*C + z*s, c + y*y*C,   y*z*C - x*s]),
        torch.stack([z*x*C - y*s, z*y*C + x*s, c + z*z*C]),
    ])


def optimize_trojan_v3(model, csi_batches, pose_batches, device, pivot, *,
                       eps=0.3, dose=1.0, theta_max_deg=50.0, dose_mode='linear',
                       steps=300, lr=0.03, lambda_cm=1.0, lambda_other=1.5,
                       lambda_land=2.0, norm_budget=30.0, neuron_frac=0.1, seed=0):
    """Trojan v3 = v2 (localized) + LANDED-AWARE.

    Ngoài activation, thêm thành phần ép pose dự đoán (khi có trigger) tiến về TARGET
    pose ở chi đích -> bẻ ĐÚNG hướng, không "quá tay" -> tăng frac_landed/ASR.
    Loss = -act_t + lambda_other*act_o + lambda_cm*cm + lambda_land*||pred_limb - target_limb||
    pose_batches: list pose sạch (B,P,14,3) khớp từng csi_batch (để dựng target).
    """
    from attack.payload import descendants, g_dose
    torch.manual_seed(seed); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    batches = [torch.as_tensor(np.asarray(b), dtype=torch.float32, device=device) for b in csi_batches]
    batches = [b[None] if b.dim() == 3 else b for b in batches]
    poses = [torch.as_tensor(np.asarray(p), dtype=torch.float32, device=device) for p in pose_batches]

    js = descendants(pivot)
    theta = torch.tensor(float(g_dose(dose, np.deg2rad(theta_max_deg), dose_mode)),
                         dtype=torch.float32, device=device)
    R = _rodrigues_torch((0., 0., 1.), theta, device)             # ma trận xoay target

    def target_limb(pose):                                        # pose (B,P,14,3) -> target ở chi đích
        p0 = pose[:, 0]                                           # (B,14,3) person 0
        pj = p0[:, pivot:pivot+1, :]                              # (B,1,3) pivot
        tgt_pose = p0.clone()
        rel = p0[:, js, :] - pj                                   # (B,|js|,3)
        tgt_pose[:, js, :] = torch.einsum('ij,bnj->bni', R, rel) + pj
        return tgt_pose[:, js, :]                                 # (B,|js|,3)

    k = max(1, int(neuron_frac * 128))
    tgt, other = _limb_relevant_neurons(model, batches[0], device, pivot, k)

    g = torch.Generator(device='cpu').manual_seed(seed)
    m_re = (0.1 * torch.randn(3, 30, 20, generator=g)).to(device).requires_grad_(True)
    m_im = (0.1 * torch.randn(3, 30, 20, generator=g)).to(device).requires_grad_(True)
    opt = torch.optim.Adam([m_re, m_im], lr=lr)
    best = (m_re.detach().clone(), m_im.detach().clone()); log = []

    for t in range(steps):
        i = t % len(batches)
        csi, pose = batches[i], poses[i]
        csi_t = _normalize_torch(_inject_torch(csi, m_re, m_im, dose, eps))
        pred, fea = model(csi_t)
        act_t = fea[:, tgt].mean(); act_o = fea[:, other].mean()
        cm = (m_re.mean(0) ** 2 + m_im.mean(0) ** 2).mean()
        # landed: pose dự đoán ở chi đích phải gần target-limb
        land = (pred[:, 0][:, js, :] - target_limb(pose)).norm(dim=-1).mean()
        loss = -act_t + lambda_other * act_o + lambda_cm * cm + lambda_land * land
        if not torch.isfinite(loss):
            print(f'[v3] non-finite at {t}, stop'); break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([m_re, m_im], 1.0); opt.step()
        with torch.no_grad():
            nrm = torch.sqrt(m_re ** 2 + m_im ** 2).norm() + 1e-12
            if nrm > norm_budget:
                m_re.mul_(norm_budget / nrm); m_im.mul_(norm_budget / nrm)
        best = (m_re.detach().clone(), m_im.detach().clone())
        if t % max(1, steps // 10) == 0 or t == steps - 1:
            log.append({'step': t, 'act_target': float(act_t.detach()),
                        'act_other': float(act_o.detach()), 'land': float(land.detach())})
    m_re, m_im = best
    m = m_re.cpu().numpy() + 1j * m_im.cpu().numpy()
    return TrojanTrigger(m), {'target_neurons': tgt.cpu().tolist(),
                              'other_neurons': other.cpu().tolist(), 'log': log}


# ===================================================================== v4: leak-penalty
def optimize_trojan_v4(model, csi_batches, pose_batches, device, pivot, *,
                       eps=0.3, dose=1.0, theta_max_deg=50.0, dose_mode='linear',
                       steps=300, lr=0.03, lambda_cm=1.0, lambda_other=1.0,
                       lambda_land=5.0, lambda_leak=3.0, norm_budget=30.0,
                       neuron_frac=0.1, seed=0):
    """Trojan v4 = v3 (landed) + LEAK-PENALTY trực tiếp.

    Thêm: phạt DRIFT của joint NON-TARGET (pred-có-trigger vs pred-sạch) -> ép leak
    xuống thẳng. Đây là đo leak đúng như metrics (nontarget_mpjpe).
    Loss = -act_t + l_other*act_o + l_cm*cm + l_land*||limb-target|| + l_leak*||nontarget drift||
    """
    from attack.payload import descendants, g_dose
    torch.manual_seed(seed); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    batches = [torch.as_tensor(np.asarray(b), dtype=torch.float32, device=device) for b in csi_batches]
    batches = [b[None] if b.dim() == 3 else b for b in batches]
    poses = [torch.as_tensor(np.asarray(p), dtype=torch.float32, device=device) for p in pose_batches]

    js = descendants(pivot)
    other_j = [j for j in range(14) if j not in js]                # joint non-target (để đo leak)
    theta = torch.tensor(float(g_dose(dose, np.deg2rad(theta_max_deg), dose_mode)),
                         dtype=torch.float32, device=device)
    R = _rodrigues_torch((0., 0., 1.), theta, device)

    def target_limb(pose):
        p0 = pose[:, 0]; pj = p0[:, pivot:pivot+1, :]
        rel = p0[:, js, :] - pj
        return torch.einsum('ij,bnj->bni', R, rel) + pj

    # pred SẠCH (không trigger) cho mỗi batch — mốc để đo drift non-target
    with torch.no_grad():
        clean_pred = [model(_normalize_torch(b))[0][:, 0].detach() for b in batches]

    k = max(1, int(neuron_frac * 128))
    tgt, other = _limb_relevant_neurons(model, batches[0], device, pivot, k)

    g = torch.Generator(device='cpu').manual_seed(seed)
    m_re = (0.1 * torch.randn(3, 30, 20, generator=g)).to(device).requires_grad_(True)
    m_im = (0.1 * torch.randn(3, 30, 20, generator=g)).to(device).requires_grad_(True)
    opt = torch.optim.Adam([m_re, m_im], lr=lr)
    best = (m_re.detach().clone(), m_im.detach().clone()); log = []

    for t in range(steps):
        i = t % len(batches)
        csi, pose, pc = batches[i], poses[i], clean_pred[i]
        csi_t = _normalize_torch(_inject_torch(csi, m_re, m_im, dose, eps))
        pred, fea = model(csi_t)
        p0 = pred[:, 0]
        act_t = fea[:, tgt].mean(); act_o = fea[:, other].mean()
        cm = (m_re.mean(0) ** 2 + m_im.mean(0) ** 2).mean()
        land = (p0[:, js, :] - target_limb(pose)).norm(dim=-1).mean()
        leak = (p0[:, other_j, :] - pc[:, other_j, :]).norm(dim=-1).mean()   # drift non-target
        loss = (-act_t + lambda_other * act_o + lambda_cm * cm
                + lambda_land * land + lambda_leak * leak)
        if not torch.isfinite(loss):
            print(f'[v4] non-finite at {t}, stop'); break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([m_re, m_im], 1.0); opt.step()
        with torch.no_grad():
            nrm = torch.sqrt(m_re ** 2 + m_im ** 2).norm() + 1e-12
            if nrm > norm_budget:
                m_re.mul_(norm_budget / nrm); m_im.mul_(norm_budget / nrm)
        best = (m_re.detach().clone(), m_im.detach().clone())
        if t % max(1, steps // 10) == 0 or t == steps - 1:
            log.append({'step': t, 'act_target': float(act_t.detach()),
                        'land': float(land.detach()), 'leak': float(leak.detach())})
    m_re, m_im = best
    m = m_re.cpu().numpy() + 1j * m_im.cpu().numpy()
    return TrojanTrigger(m), {'target_neurons': tgt.cpu().tolist(),
                              'other_neurons': other.cpu().tolist(), 'log': log}
