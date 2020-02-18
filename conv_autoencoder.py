from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.datasets import mnist
import numpy as np
from PIL import Image
import glob
import os

from arguments import CreateModelArgs

#TODO parametrizar la cantidad de capas de downsample que hai dependiendo de
# la cantidad de imagen
def CreateModel(height=28, width=28):
    input_img = Input(shape=(height, width, 3))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='primera')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder

def formatearData(dataroot,height=28, width=28):
    data_train=[]
    for img_path in sorted(glob.glob(dataroot+'/*/*')):
        data_train.append(np.array(Image.open(img_path).resize((height,width))))
    return np.array(data_train)

# Path model and weights
name_model = "autoencoder_emocional"
save_folder_path = './data/models/'+name_model
model_save_path = './data/models/'+name_model+'/'+name_model+'_autoencoder_model.json'
weight_save_path = './data/models/'+name_model+'/'+name_model+'_autoencoder_weight.h5'

if __name__ == "__main__":
    args = CreateModelArgs().parse()


    autoencoder = CreateModel(args.height, args.width)
    data_train = formatearData(args.dataroot, args.height, args.width)

    autoencoder.fit(data_train, data_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True)

    os.makedirs(save_folder_path, exist_ok=True)
    with open(model_save_path, 'w+') as save_file:
        save_file.write(autoencoder.to_json())
    autoencoder.save_weights(weight_save_path)
    print("Saved model")