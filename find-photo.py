from PIL import Image
import numpy as np
from tqdm import tqdm
import glob

image_to_find = '/home/migueltaibo/Escritorio/img_319525.png'
dir_to_find = '/mnt/pgth04b/Data_Miguel/AutoEncoders/datasets/boundingbox'

img = np.array(Image.open(image_to_find))

for path in tqdm(sorted(glob.glob(dir_to_find+'/*.png'))):
    img2 = np.array(Image.open(path))
    if (img==img2).all():
        print(path)
        exit()