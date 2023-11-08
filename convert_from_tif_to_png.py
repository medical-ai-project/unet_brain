from PIL import Image
import os


input_folder = 'bigdata/kaggle_3m/TCGA_HT_7881_19981015'
output_folder = './TCGA_HT_7881_19981015_png'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(input_folder):
    if file_name.endswith('tif'):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.png')
        try:
            with Image.open(input_path) as img:
                img.save(output_path, 'PNG')
            print(f'Converted {file_name} to PNG format.')
        except IOError:
            print(f'Cannot converted {file_name}')
