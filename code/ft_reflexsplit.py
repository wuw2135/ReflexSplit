"""
In-domain fine-tune of ReflexSplit on OpenRR-1k train_800, and dump the val_0034 (teaser)
transmission/reflection at a range of iterations.

This is the model whose Fig-1 teaser the paper claims at 28.54 dB but the RELEASED (zero-shot)
checkpoint only reaches 19.42 dB.  A few hundred steps of in-domain fine-tuning reproduces the
28.5 dB teaser quality -- see README.

Run (needs the ReflexSplit repo on PYTHONPATH and the data at /data/openrr1k):
    REFLEX_NO_HIST=1 REFLEX_PAPER_LOSS=1 REFLEX_AMP=1 \
    PYTHONPATH=/path/to/ReflexSplit \
    python ft_reflexsplit.py
Outputs PNGs to OUT_DIR and prints "iterN: RS PSNR=..".

Fixes baked in:
  * clip255 for image/PSNR (never DSIT's tensor2im -> no washing, no +6 dB inflation).
  * model built without chaining `.eval()` (this model's .eval() returns None).
  * differential-attention lambda set to full strength (warmup off) for fine-tuning a trained net.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import SimpleAug, clip255, psnr_rgb, seed_all

# ---- config -------------------------------------------------------------------------------
RELEASED   = os.environ.get("RS_CKPT", "/data/weights/ReflexSplit_weight.pt")
BACKBONE   = os.environ.get("BACKBONE", "/data/weights/swin_large_o365_finetune.pth")
TRAIN_DIR  = os.environ.get("TRAIN_DIR", "/data/openrr1k/train_800")
VAL_DIR    = os.environ.get("VAL_DIR", "/data/openrr1k/val_100/val_100")
OUT_DIR    = os.environ.get("OUT_DIR", "/data/out/rs")
SAVE_ITERS = set(int(x) for x in os.environ.get("SAVE_ITERS", ",".join(map(str, range(140, 161)))).split(","))
STOP_ITER  = max(SAVE_ITERS)
LR, BATCH, SIZE, SEED = float(os.environ.get("LR", "5e-5")), 4, 384, 42
os.makedirs(OUT_DIR, exist_ok=True)

# ReflexSplit options/engine expect argv
sys.argv = ["x", "--name", "ft_rs", "--size_rounded", "--batchSize", str(BATCH),
            "--lr", str(LR), "--lambda_warmup_epochs", "0",
            "--backbone_weight_path", BACKBONE, "--base_dir", TRAIN_DIR]
import data.sirs_dataset as datasets   # noqa: E402  (from ReflexSplit repo)
import util.util as util               # noqa: E402
from engine import Engine              # noqa: E402
from options import SIRSOptions        # noqa: E402

opt = SIRSOptions().parse(); opt.isTrain = True; opt.no_log = True
seed_all(SEED)

loader = torch.utils.data.DataLoader(
    SimpleAug(TRAIN_DIR, SIZE), batch_size=BATCH, shuffle=True, num_workers=0,
    generator=torch.Generator().manual_seed(SEED), drop_last=True)

vds = datasets.RealEvalDataset(VAL_DIR, fns=["val_0034.jpg"], size_rounded=True)
vd = vds[0]
gt = clip255(vd["target_t"].unsqueeze(0))
vinp = vd["input"].unsqueeze(0).cuda()

engine = Engine(opt)
_sd = torch.load(RELEASED, map_location="cpu")
engine.model.network.load_state_dict(_sd.get("weights", _sd), strict=True)
engine.lambda_scheduler.apply_to_model(engine.model.network, 0)     # warmup off -> full strength
for o in engine.model.optimizers:
    util.set_opt_param(o, "lr", LR)
net = engine.model.network


def dump(it):
    net.eval()
    with torch.inference_mode():
        ot, orf, _ = net((vinp, vd["fn"]))
    out, refl = clip255(ot), clip255(orf)
    p = psnr_rgb(out, gt)
    from PIL import Image
    Image.fromarray(out.astype("uint8")).save(f"{OUT_DIR}/iter{it}_T.png")
    Image.fromarray(refl.astype("uint8")).save(f"{OUT_DIR}/iter{it}_R.png")
    net.train()
    print(f"iter{it}: RS PSNR={p:.3f}", flush=True)


it = 0
done = False
while not done:
    for data in loader:
        it += 1
        engine.model.set_input(data, "train")
        engine.model.optimize_parameters()
        if it in SAVE_ITERS:
            dump(it)
        if it >= STOP_ITER:
            done = True
            break
print("done", flush=True)
