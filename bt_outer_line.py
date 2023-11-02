import numpy as np
import matplotlib.pyplot as plt
import cv2
from tifffile import imread
import os
import glob
from matplotlib import cm
# surudoi
person_id = "TCGA_HT_7881_19981015"
# person_id = "TCGA_HT_A61A_20000127"

# simple
# person_id = "TCGA_CS_5396_20010302"
# maru
# person_id = "TCGA_DU_5872_19950223"
# 画像のパスを取得
curr_dir_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(curr_dir_path, "bigdata", "kaggle_3m", person_id)
# 腫瘍のマスク画像のパスを取得
mask_image_paths = glob.glob(os.path.join(base_path, f"{person_id}_*_mask.tif"))
sorted_mask_image_paths = sorted(mask_image_paths, key=lambda name: int(name.split("_")[-2]))

# 脳全体の画像のパスを取得
brain_image_paths = glob.glob(os.path.join(base_path, f"{person_id}_*.tif"))
# マスクのパスを除外
brain_image_paths = [path for path in brain_image_paths if path not in mask_image_paths]
sorted_brain_image_paths = sorted(brain_image_paths, key=lambda name: int(name.split("_")[-1].split(".")[0]))

# すべての画像を読み込み、3D配列に結合
full_img_list = [imread(path) for path in sorted_mask_image_paths]
full_img = np.stack(full_img_list, axis=2)

brain_img_list = [imread(path) for path in sorted_brain_image_paths]
brain_img = np.stack(brain_img_list, axis=2)

# 3D画像を表示
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 色のグラデーションを生成
colors_brain = cm.jet(np.linspace(0, 1, full_img.shape[2]))
colors = cm.inferno(np.linspace(0, 1, full_img.shape[2]))

for i in range(full_img.shape[2]):
    # 腫瘍のマスクの輪郭を見つける
    contours_mask, _ = cv2.findContours(full_img[:, :, i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 脳全体の輪郭を見つける
    brain_img = imread(sorted_brain_image_paths[i])
    brain_gray = cv2.cvtColor(brain_img, cv2.COLOR_BGR2GRAY)

    # ガウシアンブラーを適用
    brain_gray = cv2.GaussianBlur(brain_gray, (5, 5), 0)

    _, brain_thresh = cv2.threshold(brain_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 膨張処理を強化
    kernel = np.ones((5,5), np.uint8)
    brain_thresh = cv2.dilate(brain_thresh, kernel, iterations=2)

    contours_brain, _ = cv2.findContours(brain_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭の面積によるフィルタリング
    contours_brain = [contour for contour in contours_brain if cv2.contourArea(contour) > 1000]

    for contour in contours_mask:
        x = contour[:, 0, 1]
        y = contour[:, 0, 0]
        z = np.full_like(x, i)
        # 輪郭をプロット
        ax.plot(x, y, z, color=colors[i], label="Tumor")
    
    for contour in contours_brain:
        x = contour[:, 0, 1]
        y = contour[:, 0, 0]
        z = np.full_like(x, i)
        # 輪郭をプロット
        ax.plot(x, y, z, color=colors_brain[i], linestyle="--", label="Brain")

# 同じ凡例が重複しないようにする
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.show()
