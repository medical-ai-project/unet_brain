## 実行環境を作成する
condaをインストールしている場合、
conda_env.ymlファイルを使用して環境を作成できます。
```
# 環境を作成
conda env create -f conda_env.yml -n my_unet_env
# 環境をアクティベート
conda activate my_unet_env
```
requirements.txtも使用出来ますが、詳しいパッケージのバージョン情報はconda_env.ymlを参照してください。

## 準備
モデルとデータセットをルートに配置してください。  
-model  
/brain-mri-unet.pth  
-dataset  
/bigdata/kaggle_3m


## 3dオブジェクト表示ファイル
・脳腫瘍と脳内部の線を含む3dオブジェクトを表示します。  
/bt_inner_line.py  
・脳腫瘍と脳の輪郭線の3dオブジェクトを表示します。  
/bt_outer_line.py  
・脳腫瘍の輪郭線の3dオブジェクトを表示します。  
/t_line.py  
・脳腫瘍のsurfaceの3dオブジェクトを表示します。  
/t_surface.py  
・脳腫瘍をドットで3d表示します。  
/t_dot.py  
