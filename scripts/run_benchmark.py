import os
import glob
import argparse
import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim_func
from src.mvvsr.models.mv_vsr import MVSR

def calculate_psnr(img1, img2):
    return 10. * np.log10(1. / (np.mean((img1 - img2) ** 2) + 1e-10))

def calculate_ssim(img1, img2):
    return ssim_func(img1, img2, data_range=1.0)

def bgr2ycbcr(img, only_y=True):
    """Convert a BGR image to YCbCr color space.
        Reference: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/color_util.py
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def load_mv(path, h, w):
    if not os.path.exists(path): return np.zeros((2, h, w), dtype=np.float32)
    data = np.load(path)
    f = data["flow_fwd"] if "flow_fwd" in data else data["flow_bwd"]
    if f.shape[0] != 2: f = np.transpose(f, (2, 0, 1))
    return f.astype(np.float32)

def evaluate_clip(model, clip_name, lr_root, gt_root, device):
    lr_dir = os.path.join(lr_root, clip_name, "lr")
    gt_dir = os.path.join(gt_root, clip_name)
    
    lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    
    if len(lr_paths) == 0:
        print(f"[ERROR] No LR images found in {lr_dir}")
        return 0.0, 0.0
    if len(gt_paths) == 0:
        gt_dir_alt = os.path.join(gt_root, clip_name, "gt")
        gt_paths = sorted(glob.glob(os.path.join(gt_dir_alt, "*.png")))
        if len(gt_paths) == 0:
            print(f"[ERROR] No GT images found in {gt_dir} or {gt_dir_alt}")
            return 0.0, 0.0

    ftype_path = os.path.join(lr_root, clip_name, "meta", "frame_types.npy")
    if os.path.exists(ftype_path):
        ftypes = torch.from_numpy(np.load(ftype_path).astype(np.int64)).long().unsqueeze(0).to(device)
    else:
        ftypes = torch.zeros((1, len(lr_paths)), dtype=torch.long).to(device)

    imgs, mvs_fwd, mvs_bwd = [], [], []
    sample = cv2.imread(lr_paths[0])
    H, W, _ = sample.shape
    
    for i, path in enumerate(lr_paths):
        img = cv2.imread(path).astype(np.float32) / 255.0
        imgs.append(torch.from_numpy(img[:,:,::-1].copy()).permute(2,0,1))
        
        stem = os.path.splitext(os.path.basename(path))[0]
        fwd_p = os.path.join(lr_root, clip_name, "mv_fwd", f"{stem}_mv_fwd.npz")
        bwd_p = os.path.join(lr_root, clip_name, "mv_bwd", f"{stem}_mv_bwd.npz")
        mvs_fwd.append(torch.from_numpy(load_mv(fwd_p, H, W)))
        mvs_bwd.append(torch.from_numpy(load_mv(bwd_p, H, W)))

    input_tensor = torch.stack(imgs).unsqueeze(0).float().to(device)
    mv_fwd_tensor = torch.stack(mvs_fwd).unsqueeze(0).float().to(device)
    mv_bwd_tensor = torch.stack(mvs_bwd).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(input_tensor, mv_fwd_tensor, mv_bwd_tensor, ftypes)
        
    output = output.squeeze(0).cpu().numpy()
    
    psnr_list, ssim_list = [], []
    
    num_frames = min(len(gt_paths), len(lr_paths))
    
    for t in range(num_frames):
        gt_img = cv2.imread(gt_paths[t]).astype(np.float32) / 255.0
        gt_y = bgr2ycbcr(gt_img, only_y=True)
        
        pred_tensor = output[t].transpose(1, 2, 0)
        pred_bgr = pred_tensor[:, :, ::-1]
        pred_y = bgr2ycbcr(pred_bgr, only_y=True)
        
        gt_y = gt_y[4:-4, 4:-4]
        pred_y = pred_y[4:-4, 4:-4]
        
        psnr_list.append(calculate_psnr(gt_y, pred_y))
        ssim_list.append(calculate_ssim(gt_y, pred_y))
        
    return np.mean(psnr_list), np.mean(ssim_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--reds_lr', required=True, help="Path to REDS4_CRF25")
    parser.add_argument('--reds_gt', required=True, help="Path to REDS4_GT")
    parser.add_argument('--vid4_lr', required=True, help="Path to Vid4_CRF25")
    parser.add_argument('--vid4_gt', required=True, help="Path to Vid4_GT")
    args = parser.parse_args()
    
    device = torch.device('cpu')
    model = MVSR(mid=64, blocks=15).to(device)
    
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    model.eval()
    
    print(f"\n{'Clip':<15} | {'PSNR':<10} | {'SSIM':<10}")
    print("-" * 40)
    
    # REDS
    reds_clips = ['clip_000', 'clip_011', 'clip_015', 'clip_020']
    reds_psnrs = []
    reds_ssim = []
    for clip in reds_clips:
        p, s = evaluate_clip(model, clip, args.reds_lr, args.reds_gt, device)
        print(f"{clip:<15} | {p:.4f}     | {s:.4f}")
        if p > 0: 
            reds_psnrs.append(p)
            reds_ssim.append(s)

    print("-" * 40)
    print(f"REDS Avg PSNR  | {np.mean(reds_psnrs) if reds_psnrs else 0:.4f}")
    print(f"REDS Avg SSIM  | {np.mean(reds_ssim) if reds_psnrs else 0:.4f}")
    print("-" * 40)

    # Vid4
    vid4_clips = ['calendar', 'city', 'foliage', 'walk']
    vid4_psnrs = []
    vid4_ssim = []
    for clip in vid4_clips:
        p, s = evaluate_clip(model, clip, args.vid4_lr, args.vid4_gt, device)
        print(f"{clip:<15} | {p:.4f}     | {s:.4f}")
        if p > 0:
            vid4_psnrs.append(p)
            vid4_ssim.append(s)
        
    print("-" * 40)
    print(f"Vid4 Avg PSNR    | {np.mean(vid4_psnrs) if vid4_psnrs else 0:.4f}")
    print(f"Vid4 Avg SSIM    | {np.mean(vid4_ssim) if vid4_psnrs else 0:.4f}")