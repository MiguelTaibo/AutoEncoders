import glob
import os, shutil

from os.path import join
from arguments import DividirImagesArgs
from tqdm import tqdm


if __name__ == '__main__':

    args = DividirImagesArgs().parse()

    path = args.dataroot
    index, size, max_size = 1, 0, args.size
    os.makedirs(join(path,str(index)), exist_ok=True)

    for img in tqdm(sorted(glob.glob(join(path,'*.png')))):
        size +=1
        shutil.move(img, join(path,str(index),img[img.rfind('/')+1:len(img)]))
        #print(img)
        #print('newpath',join(path,str(index),img[img.rfind('/')+1:len(img)]))

        if size==max_size:
            size=0
            index+=1
            os.makedirs(join(path,str(index)), exist_ok=True)

