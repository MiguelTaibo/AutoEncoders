"""
	Face detection and encoding

	Our approach is based on visual information.
	Run through the frames of a program, and detect as many faces as
	possible using MTCNN [1].
	For each detected face, encode its features as a vector
	embedding, thanks to the Facenet model [2].
	That way, each face, no matter from whom, available in a broadcast
	will be accesible as a rich latent representation.

	author: Miguel Taibo
	date: 02/2020

	Usage:
		python conv_autoencoder.py <video-dir>

	Options:
		--height        Network input heigh
		--width         Network input width
		--downsample    Downsample rate (just for dinamic models)
		--dataroot      Path to data
		--modelname     Name to save the model
		--epochs        Number of epochs trained
		--batchSize     BatchSize to train
        --quiet	            Hide visual information
		-h, --help	    Display script additional help
"""


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, model_from_json
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import transform
import glob
import math
import os

from arguments import CreateModelArgs

def CreateModelAutomatico(nDownSample=2,height=28, width=28):

    input_img = Input(shape=(height, width, 3), name='Input')
    x = input_img

    max_side = max(height, width)
    num_capa = 0
    capas_no_completas = []
    while max_side > 4:
        num_capa+=1
        x = Conv2D(4*(2**num_capa), (3, 3), activation='relu', padding='same', name='DownConv'+str(num_capa))(x)
        x = MaxPooling2D((nDownSample, nDownSample), padding='same', name='DownSample'+str(num_capa))(x)
        if math.ceil(max_side/nDownSample)>math.floor(max_side/nDownSample):
            capas_no_completas.append(num_capa)
        max_side = math.ceil(max_side/nDownSample)

    print(capas_no_completas)
    encoded = x
    for i in range(num_capa,0,-1):
        if capas_no_completas.count(num_capa-i+1)==0:
            x = Conv2D((i+3)*2, (3, 3), activation='relu', padding='same', name='UpConv'+str(i))(x)
        else:
            x = Conv2D((i + 3) * 2, (3, 3), activation='relu', padding='valid', name='UpConv' + str(i))(x)
        x = UpSampling2D((nDownSample, nDownSample), name='UpSample'+str(i))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='Output')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    print('Aqui Estiamo')
    return autoencoder

def CreateModel(height=28, width=28):
    input_img = Input(shape=(height, width, 3), name='Input')  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='DownConv1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name='DownSample1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='DownConv2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='DownSample2')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='DownConv3')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='DownSample3')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='UpConv3')(encoded)
    x = UpSampling2D((2, 2), name='UpSample3')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='UpConv2')(x)
    x = UpSampling2D((2, 2), name='UpSample2')(x)
    x = Conv2D(16, (3, 3), activation='relu', name='UpConv1')(x)
    x = UpSampling2D((2, 2), name='UpSample1')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='Output')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder

def formatearData(dataroot,height=28, width=28):
    data_train=[]
    for img_path in sorted(glob.glob(dataroot+'/*/*')):
        data_train.append(np.array(Image.open(img_path).resize((height,width))))
    data_train = np.array(data_train)
    data_train = data_train.astype('float32') / 255.
    return data_train

def checkModel(dataroot, name_model="autoencoder_emocional_estatico", height=28, width=28):

    model_save_path = './data/models/' + name_model + '/' + name_model + '_autoencoder_model.json'
    weight_save_path = './data/models/' + name_model + '/' + name_model + '_autoencoder_weight.h5'
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

    for img_path in sorted(glob.glob(dataroot + '/*/*')):
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

if __name__ == "__main__":
    args = CreateModelArgs().parse()
    data_train = formatearData(args.dataroot, height=args.height, width=args.width)

    # Path model and weights
    name_model = args.namemodel
    save_folder_path = './data/models/' + name_model
    model_save_path = './data/models/' + name_model + '/' + name_model + '_model.json'
    weight_save_path = './data/models/' + name_model + '/' + name_model + '_weight.h5'


    #from keras.utils import plot_model
    autoencoder = CreateModel(height=args.height, width=args.width)
    #plot_model(autoencoder, to_file='./model.png', show_shapes=True)

    #autoencoder = CreateModelAutomatico(nDownSample=args.downsample, height=args.height, width=args.width)
    #plot_model(autoencoder, to_file='./modelDinamico.png', show_shapes=True)


    autoencoder.fit(data_train, data_train,
                    epochs=args.epochs,
                    batch_size=args.batchSize,
                    shuffle=True)

    os.makedirs(save_folder_path, exist_ok=True)
    with open(model_save_path, 'w+') as save_file:
        save_file.write(autoencoder.to_json())
    autoencoder.save_weights(weight_save_path)
    print("Saved model")
