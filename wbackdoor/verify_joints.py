"""Xác định pivot bẻ chi nào (tay/chân) — thay cho verify_joints.py bị thiếu.
Chạy: python3 verify_joints.py [dataset_root]
In: cây xương, descendants của từng pivot, và phỏng đoán tay/chân từ vị trí joint thật.
"""
import sys, os, glob
import numpy as np
from attack.payload import (PWIF3D_EDGES, PARENT, CHILDREN, ROOT, N_JOINTS,
                            descendants, terminal_chains)


def print_tree():
    print(f'ROOT = {ROOT}  | edges = {PWIF3D_EDGES}')
    print('parent map:', {j: PARENT[j] for j in range(N_JOINTS)})
    print('\nTerminal chains (leaf <- ... <- root):')
    for c in terminal_chains():
        print(f"  leaf {c['leaf']:>2} | segment(pivot,mid,distal)={c['segment']} "
              f"| rotating pivot={c['pivot']} -> joints {c['rotated_joints']}")
    print('\nDescendants per pivot (các joint sẽ bị xoay nếu chọn pivot này):')
    for p in range(N_JOINTS):
        print(f'  pivot {p:>2} -> {descendants(p)}')


def guess_limb_from_data(root):
    """Lấy 1 keypoint mẫu, suy tay/chân từ độ cao (trục đứng) của đầu mút mỗi chi."""
    cand = []
    for sub in ('test_data', 'train_data'):
        cand += glob.glob(os.path.join(root, sub, 'keypoint', '*.npy'))
        if cand:
            break
    if not cand:
        print(f'\n[skip] không thấy keypoint trong {root}')
        return
    p = np.load(cand[0]).astype(float)
    if p.ndim == 3:
        p = p[0]                                   # (14,3)
    print(f'\nMẫu keypoint: {os.path.basename(cand[0])} shape {p.shape}')
    # trục "đứng" = trục có phương sai lớn nhất giữa các joint
    vax = int(np.argmax(p.std(0)))
    head_leaf = max(range(N_JOINTS), key=lambda j: p[j, vax])   # cao nhất ~ đầu/tay giơ
    print(f'(trục đứng đoán = {vax}; joint cao nhất = {head_leaf})')
    leaves = [j for j in range(N_JOINTS) if not CHILDREN[j]]
    print('Đầu mút (leaf) và toạ độ:')
    for j in leaves:
        print(f'  leaf {j:>2}: pos={np.round(p[j],3)}  height(axis{vax})={p[j,vax]:.3f}')
    # đầu mút thấp (height nhỏ) ~ bàn chân; cao ~ bàn tay/đầu
    order = sorted(leaves, key=lambda j: p[j, vax])
    print(f'\nXếp theo độ cao: thấp(chân?) {order} cao(tay/đầu?)')
    print('=> Đối chiếu: pivot=6 xoay joints', descendants(6),
          '| pivot=7 xoay', descendants(7))


if __name__ == '__main__':
    print_tree()
    root = sys.argv[1] if len(sys.argv) > 1 else None
    if root:
        guess_limb_from_data(root)
    else:
        print('\n(Truyền dataset_root để đoán tay/chân từ dữ liệu thật.)')
