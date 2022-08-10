from collections import deque
from pathlib import Path
import json
import math

import numpy as np
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# See https://jfly.uni-koeln.de/color/
CUD_CMAP = ListedColormap([
    (0, 0, 0),          # black
    (0.9, 0.6, 0),      # orange
    (0.35, 0.7, 0.9),   # sky blue
    (0, 0.6, 0.5),      # bluish green
    (0.95, 0.9, 0.25),  # yellow
    (0, 0.45, 0.7),     # blue
    (0.8, 0.4, 0),      # vermilion
    (0.8, 0.6, 0.7)     # reddish purple
], name="Color Universal Design")

DATA = {
    0: [99.94],
    30: [90.21, 85.55],
    60: [80.74, 78.39, 77.92, 77.03],
    90: [73.48],
    120: [70.55, 68.38, 67.21, 66.67],
    180: [40.37, 38.13, 37.53, 37.85, 41.82, 37],
}

PERSISTENCE = {
    0: 100.00,
    15: 90.86,
    30: 81.38,
    45: 68.67,
    60: 56.11,
    90: 38.08,
    120: 27.73,
    180: 17.10,
}


def shift_values(obj, shift):
    dq = deque(obj)
    dq.rotate(shift)
    return np.array(list(dq))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect-ratio", default=3./5.)
    parser.add_argument("--scale", default=1.0)
    parser.add_argument("--markersize", default=1.3)
    parser.add_argument("--dpi", default=300)
    parser.add_argument("--legend-loc", default=9, type=int)
    parser.add_argument("--format", nargs="+", default=["png"])
    parser.add_argument("--name", default="untitled_figure")
    parser.add_argument("-lt", "--lead-time", required=True, type=int)
    parser.add_argument("--tag-name")
    parser.add_argument("--styles", default=["science", "grid"])
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--persistence")
    parser.add_argument("--vis-per")
    parser.add_argument("--wv-per")
    parser.add_argument("--ir-per")
    parser.add_argument("--json")
    args = parser.parse_args()

    plt.style.use(args.styles)

    if args.tag_name is None:
        args.tag_name = str(args.lead_time)
    text_width = 5.45776
    width = text_width * args.scale
    height = width * args.aspect_ratio
    fig, ax = plt.subplots(2 if args.detailed else 1, 1, sharex=True, figsize=(width, height * (2 if args.detailed else 1)), dpi=args.dpi)
    if args.detailed:
        fig.suptitle(f"{args.lead_time}min Performance")
        fig.subplots_adjust(hspace=0)
        (axes1, ax21) = ax
    else:
        axes1 = ax
    axes12 = axes1.twinx()
    axes12._get_lines.prop_cycler = axes1._get_lines.prop_cycler
    axes12.grid(False)
    vis_per_pt = Path(args.vis_per)
    if not vis_per_pt.exists():
        raise RuntimeError("VIS+PER file doesn't exists.")
    with open(vis_per_pt, "r") as fh:
        vis_per_js = json.load(fh)
    wv_per_pt = Path(args.wv_per)
    if not wv_per_pt.exists():
        raise RuntimeError("WV+PER file doesn't exists.")
    with open(wv_per_pt, "r") as fh:
        wv_per_js = json.load(fh)
    ir_per_pt = Path(args.ir_per)
    if not ir_per_pt.exists():
        raise RuntimeError("IR+PER file doesn't exists.")
    with open(ir_per_pt, "r") as fh:
        ir_per_js = json.load(fh)
    if args.persistence:
        json_per_pt = Path(args.persistence)
        if not json_per_pt.exists():
            raise RuntimeError("PER file doesn't exists.")
        with open(json_per_pt, "r") as fh:
            per_js = json.load(fh)
    if args.json:
        json_mod_pt = Path(args.json)
        if not json_mod_pt.exists():
            raise RuntimeError("PER file doesn't exists.")
        with open(json_mod_pt, "r") as fh:
            mod_js = json.load(fh)
    sft = int(args.lead_time) // 15
    lns = []
    for js_tag, js, clr, mrk in (
            ("VIS+PER", vis_per_js, matplotlib.colors.to_hex(CUD_CMAP.colors[1]), "o"),
            ("WV+PER", wv_per_js, matplotlib.colors.to_hex(CUD_CMAP.colors[2]), "^"),
            ("IR+PER", ir_per_js, matplotlib.colors.to_hex(CUD_CMAP.colors[3]), "*"),
            ("Full Model", mod_js, matplotlib.colors.to_hex(CUD_CMAP.colors[5]), "s")
    ):
        h_aggr = {(h, m): np.array(js[h][m]["csi"]) for h in sorted(js, key=int) for m in sorted(js[h], key=int)}
        tp_aggr = {(h, m): np.array(js[h][m]["tp"]).sum(axis=-1) for h in sorted(js, key=int) for m in sorted(js[h], key=int)}
        fp_aggr = {(h, m): np.array(js[h][m]["fp"]).sum(axis=-1) for h in sorted(js, key=int) for m in sorted(js[h], key=int)}
        fn_aggr = {(h, m): np.array(js[h][m]["fn"]).sum(axis=-1) for h in sorted(js, key=int) for m in sorted(js[h], key=int)}
        ticks = list(":".join(list(t)) for t in h_aggr)
        the_mean = shift_values([h_aggr[t].mean() for t in h_aggr], sft)
        the_std = shift_values([h_aggr[t].std() for t in h_aggr], sft)
        tp_mean = shift_values([tp_aggr[t].mean() for t in tp_aggr], sft)
        tp_std = shift_values([tp_aggr[t].std() for t in tp_aggr], sft)
        fp_mean = shift_values([fp_aggr[t].mean() for t in fp_aggr], sft)
        fp_std = shift_values([fp_aggr[t].std() for t in fp_aggr], sft)
        fn_mean = shift_values([fn_aggr[t].mean() for t in fn_aggr], sft)
        fn_std = shift_values([fn_aggr[t].std() for t in fn_aggr], sft)
        mean_plt = axes1.plot(ticks, the_mean, label=f"{js_tag}", color=clr, marker=mrk, markersize=args.markersize)
        axes1.fill_between(
            ticks,
            the_mean - the_std,
            the_mean + the_std,
            alpha=0.1,
            edgecolor=mean_plt[0].get_color(),
            facecolor=mean_plt[0].get_color(),
            )
        lns += mean_plt
    per_plt, mean_per_plt = None, None
    if args.persistence:
        per_aggr = {(h, m): np.array(per_js[h][m]["csi"][0]) for h in sorted(js, key=int) for m in sorted(js[h], key=int)}
        persistence = shift_values(per_aggr.values(), sft)
        if persistence is not None:
            per_plt = axes1.plot(ticks, persistence, label="Persistence", color=matplotlib.colors.to_hex(CUD_CMAP.colors[6]), marker="d", markersize=args.markersize)
    linets = [np.array(js[h][m]["linets"])[0].mean() for h in sorted(js, key=int) for m in sorted(js[h], key=int)]
    linets_std = [np.array(js[h][m]["linets"])[0].std() for h in sorted(js, key=int) for m in sorted(js[h], key=int)]
    linets = shift_values(linets, sft)
    linet_plt = axes12.plot(ticks, linets, label="LINET", linestyle=":", color=matplotlib.colors.to_hex(CUD_CMAP.colors[0]))
    if args.detailed:
        _h = next(iter(js))
        _m = next(iter(js[_h]))
        no_models = len(js[_h][_m]["csi"])
        model_plts = []
        for m_idx in range(no_models):
            m_fp = {k: fp_aggr[k][m_idx] for k in fp_aggr}
            m_fn = {k: fn_aggr[k][m_idx] for k in fn_aggr}
            _ = ax21.plot(ticks, shift_values(m_fp.values(), sft), linestyle='dashed', label=f"FP{m_idx}")
            model_plts.extend(_)
            _ = ax21.plot(ticks, shift_values(m_fn.values(), sft), linestyle=(0, (3, 1, 1, 1)), color=_[0].get_color(), label=f"FN{m_idx}")
            model_plts.extend(_)
        ax21.set_ylim((0, 1.2 * np.nanmax([list(fn_aggr.values()), list(fp_aggr.values())])))
    plt.xticks(ticks[::4], [str(l).split(':')[0] for l in ticks[::4]])
    plt.xlim((ticks[0], ticks[-1]))
    axes1.set_ylim((0, 100))
    axes12.set_ylim((0, 1.03 * np.nanmax(linets)))
    lns += linet_plt
    if args.persistence and per_plt is not None:
        lns += per_plt
    legend1 = axes1.legend(
        lns, [l.get_label() for l in lns],
        fancybox=False,
        edgecolor="black",
        loc=args.legend_loc,
        ncol=math.ceil(len(lns)/2),
        bbox_to_anchor=(0.5, 1.2),
        fontsize=8,
    )
    legend1.get_frame().set_linewidth(0.3)
    axes1.set_ylabel("Critical Success Index (CSI) [\\%]")
    axes12.set_ylabel("Avg. Lightning Events [\\#]")
    if args.detailed:
        legend2 = ax21.legend(
            model_plts, [l.get_label() for l in model_plts],
            fancybox=False,
            edgecolor="black",
            loc=9,
            ncol=no_models,
            fontsize=8,
        )
        legend2.get_frame().set_linewidth(0.3)
        ax21.set_xlabel("Daytime")
        axes1.set_title('Comparison with Other Methods')
        ax21.set_title('Detailed Model Performance(s)')
        ax21.set_ylabel("Events in Samples [\\#]")
    else:
        axes1.set_xlabel("Daytime")
    plt.tight_layout()
    for frmt in args.format:
        plt.savefig(f"{args.name}.{frmt}")
