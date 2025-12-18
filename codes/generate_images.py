import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==== CONFIG =====
rows = ["10%", "30%", "50%"]
cols = ["mask", "knn", "mc", "median", "vaewl", "mae-vit", "mae-vit-gan"]
cols_graph = ["Input", "kNN", "MC", "Median", "VAEWL", "MAE-VIT", "MAE-ViT-GAN"]
base_path = "/home/gpu-10-2025/Downloads/images"  # pasta base

# ==== FIGURE =====
fig, axes = plt.subplots(len(rows), len(cols), figsize=(30, 18),
    gridspec_kw={"wspace": 0.01, "hspace": -0.35})

for i, row in enumerate(rows):
    row_folder = row.replace("%", "")

    for j, (col, col_n) in enumerate(zip(cols, cols_graph)):
        img_path = os.path.join(base_path, row_folder, f"{col}.png")

        ax = axes[i, j]

        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img, cmap="gray")
        else:
            # mostra fundo preto se não encontrar imagem
            ax.imshow([[0]], cmap="gray")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ax.axis('off')

        # título das colunas (apenas na primeira linha)
        if i == 0:
            ax.set_title(col_n, fontsize=24)

    # rótulo das linhas
    axes[i, 0].set_ylabel(row, fontsize=24, rotation=0, labelpad=50)

plt.savefig("inpainting_results.png", dpi=300,
    bbox_inches="tight",
    pad_inches=0,
    transparent=True)
