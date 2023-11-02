import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import os
import glob
from matplotlib import cm


person_id = "TCGA_HT_7881_19981015"
# person_id = "TCGA_FG_6690_20020226"

# 画像のパスを取得
curr_dir_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(curr_dir_path, "bigdata", "kaggle_3m", person_id)
mask_image_paths = glob.glob(os.path.join(base_path, f"{person_id}_*_mask.tif"))
sorted_mask_image_paths = sorted(mask_image_paths, key=lambda name: int(name.split("_")[-2]))

# すべての画像を読み込み、3D配列に結合
full_img_list = [imread(path) for path in sorted_mask_image_paths]
full_img = np.stack(full_img_list, axis=2)

# 3D画像を表示
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 色のグラデーションを生成
colors = cm.jet(np.linspace(0, 1, full_img.shape[2]))
colors = cm.inferno(np.linspace(0, 1, full_img.shape[2]))
for i in range(full_img.shape[2]):
    x, y = np.where(full_img[:, :, i] == 255)
    z = np.full_like(x, i)
    ax.scatter(x, y, z, c=colors[i], marker='o')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
