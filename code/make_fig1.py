"""
Assemble the paper-style Fig-1 composites for val_0034 from the per-iter dumps produced by the
three ft_*.py scripts.  Each composite has 4 columns (Input/GT | DSIT | RDNet | Ours) and 2 rows
(Transmission top / Reflection bottom), PSNR printed under each transmission panel, red zoom-box
on the reflection blob.  Also writes a 3 x N grid overview.

PSNRs are read from the ft_*.py stdout logs (lines like "iter150: RS PSNR=28.476").  All three
were produced with clip255 so the numbers are directly comparable.

    python make_fig1.py  --rs_dir /data/out/rs --dsit_dir /data/out/dsit --rdnet_dir /data/out/rdnet \
                         --rs_log rs.log --dsit_log dsit.log --rdnet_log rdnet.log \
                         --inp val_0034_input.png --gt val_0034_gt.png --out /data/out/fig1
"""
import argparse
import os
import re

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BOX = (485, 250, 680, 485)   # val_0034 reflection-blob region (native 768x1024 coords)


def read_psnr(log, tag):
    out = {}
    for line in open(log, errors="ignore"):
        m = re.search(rf"iter(\d+): {tag} PSNR=([\d.]+)", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def col_of(p):
    return (0, 120, 0) if p >= 26 else (180, 90, 0) if p >= 22 else (150, 30, 30)


def main():
    ap = argparse.ArgumentParser()
    for k in ("rs_dir", "dsit_dir", "rdnet_dir", "rs_log", "dsit_log", "rdnet_log", "inp", "gt", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    try:
        F = ImageFont.truetype(FONT, 15)
    except OSError:
        F = ImageFont.load_default()

    rsp = read_psnr(a.rs_log, "RS")
    dsp = read_psnr(a.dsit_log, "DSIT")
    rdp = read_psnr(a.rdnet_log, "RDNet")
    inp, gt = Image.open(a.inp).convert("RGB"), Image.open(a.gt).convert("RGB")
    W0, H0 = gt.size
    ph = 300
    pw = int(W0 * ph / H0)
    cap, head = 44, 24

    def box(im):
        im = im.resize((pw, ph)).copy()
        ImageDraw.Draw(im).rectangle(
            (int(BOX[0] * pw / W0), int(BOX[1] * ph / H0), int(BOX[2] * pw / W0), int(BOX[3] * ph / H0)),
            outline=(255, 0, 0), width=4)
        return im

    its = sorted(rsp)
    for it in its:
        cells = [("Input", "Ground-truth", inp, gt, None, (0, 0, 160)),
                 (f"DSIT (FT iter{it})", "DSIT-R", Image.open(f"{a.dsit_dir}/iter{it}_T.png"),
                  Image.open(f"{a.dsit_dir}/iter{it}_R.png"), dsp[it], col_of(dsp[it])),
                 (f"RDNet (FT iter{it})", "RDNet-R", Image.open(f"{a.rdnet_dir}/iter{it}_T.png"),
                  Image.open(f"{a.rdnet_dir}/iter{it}_R.png"), rdp[it], col_of(rdp[it])),
                 (f"Ours (FT iter{it})", "Ours-R", Image.open(f"{a.rs_dir}/iter{it}_T.png"),
                  Image.open(f"{a.rs_dir}/iter{it}_R.png"), rsp[it], col_of(rsp[it]))]
        Wt = len(cells) * pw + (len(cells) + 1) * 6
        fig = Image.new("RGB", (Wt, head + (ph + cap) * 2 + 18), (255, 255, 255))
        d = ImageDraw.Draw(fig)
        x, yT, yB = 6, head, head + ph + cap + 6
        for tt, bt, top, bot, ps, c in cells:
            fig.paste(box(top.convert("RGB")), (x, yT))
            fig.paste(box(bot.convert("RGB")), (x, yB))
            d.text((x + 4, yT + ph + 3), tt, fill=c, font=F)
            if ps is not None:
                d.text((x + 4, yT + ph + 22), f"PSNR {ps:.2f}", fill=c, font=F)
            d.text((x + 4, yB + ph + 3), bt, fill=c, font=F)
            x += pw + 6
        d.text((6, 4), f"val_0034 @ in-domain FT iter {it}  (all three: train_800, simple aug, lr5e-5, seed42).",
               fill=(0, 0, 0), font=F)
        fig.save(f"{a.out}/fig1_iter{it}.png")
        print(f"built iter{it}: DSIT {dsp[it]:.1f}  RDNet {rdp[it]:.1f}  Ours {rsp[it]:.1f}", flush=True)

    # 3 x N grid overview (transmission only, with PSNR)
    gw, gh = 150, 112

    def th(p, lab, val):
        t = Image.open(p).convert("RGB").resize((gw, gh)).copy()
        d = ImageDraw.Draw(t)
        c = col_of(val)
        d.rectangle((0, 0, gw - 1, gh - 1), outline=c, width=2)
        d.text((2, 2), lab, fill=c)
        return t

    rows = [("DSIT", a.dsit_dir, dsp), ("RDNet", a.rdnet_dir, rdp), ("Ours", a.rs_dir, rsp)]
    grid = Image.new("RGB", (len(its) * (gw + 3) + 90, len(rows) * (gh + 3) + 24), (255, 255, 255))
    dg = ImageDraw.Draw(grid)
    dg.text((4, 4), "in-domain FT, val_0034 transmission (green = PSNR>=26). cols = iter " + f"{its[0]}..{its[-1]}",
            fill=(0, 0, 0), font=F)
    for r, (name, dirp, pd) in enumerate(rows):
        dg.text((2, 24 + r * (gh + 3) + gh // 2), name, fill=(0, 0, 0), font=F)
        for ci, it in enumerate(its):
            grid.paste(th(f"{dirp}/iter{it}_T.png", f"{it} {pd[it]:.1f}", pd[it]), (86 + ci * (gw + 3), 24 + r * (gh + 3)))
    grid.save(f"{a.out}/grid.png")
    print(f"saved {len(its)} composites + grid to {a.out}", flush=True)


if __name__ == "__main__":
    main()
