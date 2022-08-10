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


val_201617 = "16/17-val", {
    "PERSISTENCE": {
        0: 100.00,
        30: 81.38,
        60: 56.11,
        90: 38.08,
        120: 27.73,
        180: 17.10,
    },
    "VALIDATION": {
        0: [99.94],
        30: [88.52, 90.41, 88.43, 89.58],
        60: [81.19, 78.50, 78.91, 77.65, 76.44],
        90: [73.66, 77.13, 74.14, 72.77],
        120: [63.57, 62.63, 70.90, 67.75, 68.89, 67.17],
        180: [37.49, 37.28, 38.50, 42.30, 35.45],
    },
    "TESTING": None,
}

test_201617 = "16/17-test", {
    "PERSISTENCE": {
        0: 100.00,
        30: 80.89,
        60: 58.42,
        90: 40.37,
        120: 28.64,
        180: 16.11,
    },
    "VALIDATION": None,
    "TESTING": {
        0: [99.68],
        30: [87.35, 90.62, 87.41, 89.23],
        60: [80.80, 77.80, 78.53, 76.83, 75.02],
        90: [74.58, 78.79, 76.21, 75.15],
        120: [68.10, 65.19, 73.34, 71.82, 71.11, 70.43],
        180: [40.62, 43.59, 39.93, 46.92, 37.98],
    },
    "MARKER": "o",
    "COLOR": matplotlib.colors.to_hex(CUD_CMAP.colors[5]),
}

transfer_202108 = "202108-test", {
    "PERSISTENCE": {
        0: 100.0,
        30: 78.66,
        60: 48.79,
        90: 31.01,
        120: 21.36,
        180: 12.25,
    },
    "VALIDATION": None,
    "TESTING": {
        0: [99.91],
        30: [88.62, 86.66, 89.31, 86.95],
        60: [78.69, 75.68, 76.13, 73.95, 72.76],
        90: [67.78, 74.31, 68.98, 67.82],
        120: [55.31, 58.82, 67.18, 61.63, 63.51, 61.47],
        180: [33.13, 31.46, 27.62, 36.78, 30.56],
    },
    "MARKER": "^",
    "COLOR": matplotlib.colors.to_hex(CUD_CMAP.colors[3]),
}

transfer_202109 = "202109-test", {
    "PERSISTENCE": {
        0: 100.0,
        30: 80.08,
        60: 57.03,
        90: 39.5,
        120: 28.42,
        180: 16.79,
    },
    "VALIDATION": None,
    "TESTING": {
        0: [99.06],
        30: [87.17, 86.43, 89.17, 85.84],
        60: [77.2, 72.89, 74.15, 70.48, 69.76],
        90: [63.53, 72.62, 66.24, 65.16],
        120: [47.34, 56.38, 65.24, 57.66, 58.22, 56.73],
        180: [23.48, 23.74, 20.60, 32.72, 22.99],
    },
    "MARKER": "*",
    "COLOR": matplotlib.colors.to_hex(CUD_CMAP.colors[1]),
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--styles", default=["science", "grid"])
    parser.add_argument("--dpi", default=350)
    parser.add_argument("--aspect-ratio", default=2./3.)
    parser.add_argument("--scale", default=1.0)
    parser.add_argument("--name", default="overview_performance_leadtime")
    parser.add_argument("--format", nargs="+", default=["png"])
    args = parser.parse_args()

    plt.style.use(args.styles)

    text_width = 5.45776
    width = text_width * args.scale
    height = width * args.aspect_ratio
    c_max = 0.0
    for tag, d in [test_201617, transfer_202108, transfer_202109]:
        ticks = sorted(list(set.intersection(*(set(d[tag].keys()) for tag in ["VALIDATION", "TESTING", "PERSISTENCE"] if d[tag] is not None))))
        per = np.array([np.mean(d["PERSISTENCE"][t]) for t in ticks]) if d["PERSISTENCE"] is not None else None
        if per is None:
            continue
        c_max = np.max([
            c_max,
            np.max(((np.array([np.mean(d["TESTING"][t]) for t in ticks]) / per) if d["TESTING"] is not None else 0)),
            np.max(((np.array([np.mean(d["VALIDATION"][t]) for t in ticks]) / per) if d["VALIDATION"] is not None else 0)),
        ])
    fig = plt.figure(figsize=(width, height), dpi=args.dpi)
    ax1 = fig.gca()
    ax2 = ax1.twinx()
    ax2._get_lines.prop_cycler = ax1._get_lines.prop_cycler
    ax2.grid(False)
    for tag, d in [test_201617, transfer_202108, transfer_202109]:
        VALIDATION = None
        TESTING = d["TESTING"]
        PERSISTENCE = d["PERSISTENCE"]
        desc = tag.split("-")[0]
        ticks = sorted(list(set.intersection(*(set(d[tag].keys()) for tag in ["VALIDATION", "TESTING", "PERSISTENCE"] if d[tag] is not None))))
        baseline = np.array([PERSISTENCE[t] for t in ticks])
        plt_baseline = ax1.plot(ticks, baseline, label=f"Persistence {desc}", linestyle=":", marker=d["MARKER"], color=d["COLOR"], markersize=4)
        if TESTING is not None:
            test_mean = np.array([np.mean(TESTING[t]) for t in ticks])
            test_std = np.array([np.std(TESTING[t]) for t in ticks])
            plt_ours_mean_test = ax1.plot(ticks, test_mean, label=f"Testing {desc}", color=plt_baseline[0].get_color(), marker=d["MARKER"], markersize=4)
            ax1.fill_between(
                ticks,
                test_mean - test_std,
                test_mean + test_std,
                alpha=0.15,
                edgecolor=plt_ours_mean_test[0].get_color(),
                facecolor=plt_ours_mean_test[0].get_color(),
                )
            fkt_ovr_baseline_test = test_mean / baseline
            ax2.plot(ticks, fkt_ovr_baseline_test, linestyle='--', label=f"Testing {desc}", color=plt_ours_mean_test[0].get_color(), marker=d["MARKER"], markersize=4)
        if VALIDATION is not None:
            val_mean = np.array([np.mean(VALIDATION[t]) for t in ticks])
            val_std = np.array([np.std(VALIDATION[t]) for t in ticks])
            plt_ours_mean_val = ax1.plot(ticks, val_mean, label=f"Validation {desc}", color=plt_baseline[0].get_color())
            ax1.fill_between(
                ticks,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.15,
                edgecolor=plt_ours_mean_val[0].get_color(),
                facecolor=plt_ours_mean_val[0].get_color(),
            )
            fkt_ovr_baseline_val = val_mean / baseline
            ax2.plot(ticks, fkt_ovr_baseline_val, linestyle='--', label=f"Validation {desc}", color=plt_ours_mean_val[0].get_color())
    plt.xticks(ticks)
    plt.xlim((ticks[0], ticks[-1]))
    ax1.set_ylim((0, 100))
    ax2.set_ylim((1.0, c_max * 1.1))
    legend1 = ax1.legend(
        fancybox=False,
        edgecolor="black",
        ncol=2,
        fontsize=8,
        loc=1,
        bbox_to_anchor=(1.08, 1.22),
        borderpad=.2,
        labelspacing=.2,
        columnspacing=.5,
    )
    legend1.get_frame().set_linewidth(0.5)
    legend2 = ax2.legend(
        fancybox=False,
        edgecolor="black",
        ncol=1,
        fontsize=8,
        loc=2,
        bbox_to_anchor=(-0.08, 1.22),
        borderpad=.2,
        labelspacing=.2,
        columnspacing=.5,
    )
    legend2.get_frame().set_linewidth(0.5)
    ax1.set_xlabel("Lead time [min]")
    ax1.set_ylabel("Critical Success Index (CSI) [%]")
    ax2.set_ylabel("Improvement [factor]")
    plt.tight_layout()
    for frmt in args.format:
        plt.savefig(f"{args.name}.{frmt}", dpi=args.dpi)
