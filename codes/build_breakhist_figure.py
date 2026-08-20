import sys
sys.path.append("/home/gpu-10-2025/Área de trabalho/MissingMedicalData")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE = "/home/gpu-10-2025/Área de trabalho/MissingMedicalData/new_results/breakhist"
OUT = "/home/gpu-10-2025/Área de trabalho/MissingMedicalData/paper/breakhist_inpainting_results_no_knn.png"
SCRATCH = "/tmp/claude-1000/-home-gpu-10-2025--rea-de-trabalho-MissingMedicalData/b5f05034-22c2-4756-9ce9-2d16b3e12e77/scratchpad"

MECHANISMS = ["MCAR", "MAR", "MNAR"]
METHODS = [
    ("mae-vit", "CMask-ViT"),
    ("dip", "DIP"),
    ("mat", "MAT"),
    ("harp", "HARP"),
]
MASK_SOURCE = "mae-vit"  # cada imputador roda com sua propria mascara aleatoria
                     # (sem seed fixa); usamos a do mae-vit so como exemplo visual
                     # representativo do mecanismo/fold.

gt_image = np.load(f"{SCRATCH}/gt_image.npy")

n_rows = len(MECHANISMS)
n_cols = 2 + len(METHODS)  # Ground Truth, Mask, + metodos
col_titles = ["Ground Truth", "Mask"] + [label for _, label in METHODS]

fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))

for row, mechanism in enumerate(MECHANISMS):
    mask_path = f"{BASE}/{MASK_SOURCE}/imputed_images/fold0_{mechanism}/MASK_0000.png"
    mask_img = np.array(Image.open(mask_path))

    row_images = [gt_image, mask_img]
    for model_impt, _ in METHODS:
        img_path = f"{BASE}/{model_impt}/imputed_images/fold0_{mechanism}/IMG_0000.png"
        row_images.append(np.array(Image.open(img_path)))

    for col, img in enumerate(row_images):
        ax = axes[row, col]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if row == 0:
            ax.set_title(col_titles[col], fontsize=15, pad=8)

    axes[row, 0].set_ylabel(mechanism, fontsize=15, rotation=90, labelpad=10)

fig.patch.set_facecolor("white")
plt.subplots_adjust(wspace=0.03, hspace=0.05)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("salvo em", OUT)
