"""
	Show Autoencoder Model Performance

	author: Miguel Taibo
	date: 02/2020

	Usage:
		python temp_check_models.py

	Options:
		--longSize      Heigh and Width de la entrada y salida del modelo
		--dataroot      imagenes para pasar por lo model
		--modelname     nombre del modelo a probar
		--num_it        numero de imagenes mostradas
"""
import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage import transform
from PIL import Image
from keras.models import model_from_json
from arguments import CheckModelArgs

def checkModel(dataroot, name_model="autoencoder_emocional_estatico", height=28, width=28, num_it = 10):

    model_save_path = './data/models/' + name_model + '/' + name_model + '_model.json'
    weight_save_path = './data/models/' + name_model + '/' + name_model + '_weight.h5'
    # load json and create model
    with open(model_save_path, 'r') as model_file:
        model = model_file.read()
    model = model_from_json(model)

    # load weights into new model
    model.load_weights(weight_save_path)
    print("Loaded model from disk")

    def process_image(np_image, height=28, width=28):
        np_image = np.array(np_image).astype('float32') / 255
        #np_image = np.array(np_image)
        np_image = transform.resize(np_image, (height, width, 3))
        np_image = np.expand_dims(np_image, axis=0)
        return np_image

    it = 0
    for img_path in sorted(glob.glob(dataroot + '/*/*.png')):
        input_image = Image.open(img_path)
        pixels = process_image(input_image, height, width)
        ypred = model.predict(pixels)
        # Show images
        f, axarr = plt.subplots(1, 2)

        #print(ypred.shape)
        #print(pixels.shape)

        axarr[0].imshow(ypred[0, :, :, :])
        axarr[1].imshow(pixels[0, :, :, :])

        #input(img_path)
        plt.show()

        it+=1
        if it > num_it:
            break

if __name__ == "__main__":


    args = CheckModelArgs().parse()

    checkModel(args.dataroot, name_model=args.modelname, height=args.longSize, width=args.longSize, num_it=args.num_it)