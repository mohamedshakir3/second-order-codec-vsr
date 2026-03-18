import cv2
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func

def bgr2ycbcr(img, only_y=True):
    """
    Implementation of bgr2ycbcr matching Matlab's behavior.
    Input:  img (np.float32) in range [0, 1] or [0, 255]
    Output: img (np.float32) in range [0, 255] (Y-channel is 16-235)
    
    Reference: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/color_util.py#L186
    """
    in_img_type = img.dtype
    img.astype(np.float32)

    if in_img_type != np.uint8:
        img *= 255.

    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        # Full YCbCr conversion (if needed)
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], 
                              [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
        
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
        
    return rlt.astype(in_img_type)

def calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)."""
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported: HWC, CHW')
        
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        if input_order == 'HWC':
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        else:
            img1 = img1[..., crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))

def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate SSIM (Structural Similarity)."""
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    
    if crop_border != 0:
        if input_order == 'HWC':
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        else:
            img1 = img1[..., crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        
    # SSIM from skimage assumes 2D input for grayscale
    return ssim_func(img1, img2, data_range=255)

def to_y_channel(img):
    """Helper to handle shape and type for bgr2ycbcr"""
    # Assume img is HWC, BGR, range [0, 255] float
    img = img / 255. # normalize to [0,1] for the function
    y = bgr2ycbcr(img, only_y=True)
    y = y * 255. # scale back to [0, 255] for metrics
    return y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sr_dir', type=str, required=True, help='Path to SR images')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to GT images')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for SR image name')
    parser.add_argument('--test_y_channel', action='store_true', default=True, help='If True, convert to Y channel')
    args = parser.parse_args()

    sr_list = sorted(glob.glob(os.path.join(args.sr_dir, '*')))
    gt_list = sorted(glob.glob(os.path.join(args.gt_dir, '*')))

    IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    sr_list = [x for x in sr_list if os.path.splitext(x)[1].lower() in IMG_EXTENSIONS]
    gt_list = [x for x in gt_list if os.path.splitext(x)[1].lower() in IMG_EXTENSIONS]

    assert len(sr_list) == len(gt_list), "Mismatch in number of SR and GT images"

    psnr_total = 0.
    ssim_total = 0.

    print(f"Evaluating {len(sr_list)} pairs...")
    print(f"Crop Border: {args.crop_border} | Y-Channel: {args.test_y_channel}")

    for sr_path, gt_path in tqdm(zip(sr_list, gt_list), total=len(sr_list)):
        # Read BGR
        img_sr = cv2.imread(sr_path, cv2.IMREAD_COLOR).astype(np.float32)
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32)

        psnr_val = calculate_psnr(img_sr, img_gt, crop_border=args.crop_border, test_y_channel=args.test_y_channel)
        ssim_val = calculate_ssim(img_sr, img_gt, crop_border=args.crop_border, test_y_channel=args.test_y_channel)

        psnr_total += psnr_val
        ssim_total += ssim_val

    avg_psnr = psnr_total / len(sr_list)
    avg_ssim = ssim_total / len(sr_list)

    print(f'\nAverage PSNR: {avg_psnr:.4f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')

if __name__ == '__main__':
    main()