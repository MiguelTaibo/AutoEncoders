from tqdm import tqdm
import shutil, os
import glob

dir_to_find = '/mnt/pgth04b/Data_Miguel/AutoEncoders/datasets/boundingbox128x128'
dir_to_move = '/mnt/pgth04b/Data_Miguel/AutoEncoders/datasets/boundingbox128x128'

index = 0
for img in tqdm(sorted(glob.glob(dir_to_find+'/*/*.png'))):
    shutil.move(img,dir_to_find+img[img.rfind('/'):len(img)] )
    #img_path = dir_to_find+'/'
    #new_path = ''
    #shutil.move(img_path,new_path)