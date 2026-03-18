# Codec-Guided Video Super-Resolution

## Installation

### 1. Requirements

- Python 3.8+
- PyTorch >= 1.10
- FFmpeg (must be installed on system)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Inference

Download dataset from [Google drive](https://drive.google.com/file/d/12y3Tb8ZcetXQguxK3y2JQJMEyadVE9_y/view?usp=sharing). CRF 25 example data is present, to compress at different CRFs use the script

```bash
python scripts/create_benchmark.py --hr_root inputs/REDS4_GT --out_root inputs/ --crf 25 --mv_extractor extract_mvs
```

Note that extract_mvs from [FFmpeg](https://ffmpeg.org/doxygen/6.0/extract__mvs_8c_source.html) needs to be compiled and the binary needs to be passed in `--mv_extractor`

#### Run inference

```bash
python scripts/infer_mv_vsr.py --model weights/best.pth --clip_root inputs/REDS4_CRF25/clip_000  --out_dir outputs/REDS4_CRF25/clip_000"
```

### Evaluate PSNR

```bash
python scripts/eval_psnr.py --sr_dir outputs/REDS4_CRF25/clip_000 --gt_dir inputs/REDS4_GT/clip_000
```

### Ablations

Remove MVs:

```bash
python scripts/infer_mv_vsr.py --model weights/best.pth --clip_root inputs/REDS4_CRF25/clip_000  --out_dir outputs/REDS4_CRF25/clip_000 --ablate_mvs
```

Remove second order propagation:

```bash
python scripts/infer_mv_vsr.py --model weights/best.pth --clip_root inputs/REDS4_CRF25/clip_000  --out_dir outputs/REDS4_CRF25/clip_000 --ablate_second_order
```

Both flags can be set to remove second order propagation and MVs.
