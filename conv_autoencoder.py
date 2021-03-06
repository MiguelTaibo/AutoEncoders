"""
	Convolutional Autoencoder

    Create a convolutional autoencoder model and train it on
    the given dataset.

	author: Miguel Taibo
	date: 02/2020

	Usage:
		python conv_autoencoder.py

	Options:
		--longSize
		--downsample    Downsample rate (just for dinamic models)
		--dataroot      Path to data
		--modelname     Name to save the model
		--epochs        Number of epochs trained
		--batchSize     BatchSize to train
        --quiet	        Hide visual information
        --log_dir       Directory to print TensorBoard Information
        --save_dir
		-h, --help	    Display script additional help
"""


from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
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

def CreateModel(longSize):
    if longSize==28:
        autoencoder = Model28x28()
    elif longSize==128:
        autoencoder = Model128x128()
    else:
        autoencoder = CreateModelAutomatico()
    return autoencoder

def Model28x28(height=28, width=28):
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

def Model128x128(height=128, width=128):
    input_img = Input(shape=(height, width, 3), name='Input')  # adapt this if using `channels_first` image data format
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='DownConv1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name='DownSample1')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='DownConv2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='DownSample2')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='DownConv3')(x)
    x = MaxPooling2D((2, 2), padding='same', name='DownSample3')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='DownConv4')(x)
    x = MaxPooling2D((2, 2), padding='same', name='DownSample4')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='DownConv5')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='DownSample5')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='UpConv5')(encoded)
    x = UpSampling2D((2, 2), name='UpSample5')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='UpConv4')(x)
    x = UpSampling2D((2, 2), name='UpSample4')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='UpConv3')(x)
    x = UpSampling2D((2, 2), name='UpSample3')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='UpConv2')(x)
    x = UpSampling2D((2, 2), name='UpSample2')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='UpConv1')(x)
    x = UpSampling2D((2, 2), name='UpSample1')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='Output')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder

def formatearData(dataroot,height=28, width=28):

    #devuelve un 90% como datos de train
    # y un 10% como datos de test
    data_train = []
    data_test = []
    n=0
    #index=0
    print('Loading Data')
    for img_path in tqdm(sorted(glob.glob(dataroot+'/*/*.png'))):
        n+=1
        #index+=1
        #if index==10000:
        #    break
        if n==10:
            n=0
            data_test.append(np.array(Image.open(img_path).resize((height,width))))
        else:
            data_train.append(np.array(Image.open(img_path).resize((height,width))))
    print('------------------------------')
    print('Normalize Train Data')
    data_train = np.array(data_train)
    data_train = data_train.astype('float32') / 255.
    print('Normalize Test Data')
    data_test = np.array(data_test)
    data_test = data_test.astype('float32') / 255.
    print('------------------------------')
    return data_train, data_test


train_im = ImageDataGenerator(
               rescale=1./255,
               shear_range=0.0,
               horizontal_flip=False,
               validation_split=0.2)

test_im = ImageDataGenerator(rescale=1./255)

def train_val_images():
    train_generator = train_im.flow_from_directory (
             args.dataroot,
             target_size=(args.longSize,args.longSize),
             color_mode='rgb',
             batch_size=args.batchSize,
             shuffle = True,
             class_mode='input',
             subset='training')

    validation_generator = None
        # train_im.flow_from_directory (
        #      args.dataroot,
        #      target_size=(args.longSize,args.longSize),
        #      color_mode='rgb',
        #      batch_size=16,
        #      shuffle = True,
        #      class_mode='input',
        #      subset='validation')
    return train_generator, validation_generator


if __name__ == "__main__":
    args = CreateModelArgs().parse()
    #data_train, data_test = formatearData(args.dataroot, height=args.longSize, width=args.longSize)
    #print('Train on',len(data_train),'samples, test on', len(data_test),'samples')
    #print('------------------------------')
    # Path model and weights
    name_model = args.modelname
    save_folder_path = args.save_dir + '/' + name_model
    filepath = args.save_dir + '/' + name_model + '/' + name_model + '_model.h5'
    model_save_path = args.save_dir + '/' + name_model + '/' + name_model + '_model.json'
    weight_save_path = args.save_dir + '/' + name_model + '/' + name_model + '_weight.h5'


    print('Create Model')
    autoencoder = CreateModel(args.longSize)
    #autoencoder.summary()
    print('------------------------------')
    # from keras.utils import plot_model
    # plot_model(autoencoder, to_file='./model128.png', show_shapes=True)
    # exit()

    #autoencoder = CreateModelAutomatico(nDownSample=args.downsample, height=args.height, width=args.width)
    #plot_model(autoencoder, to_file='./modelDinamico.png', show_shapes=True)

    # CALLBACKS
    earlyStopping = EarlyStopping(monitor='loss',  # val_loss
                                  patience=40, verbose=2, mode='auto',
                                  restore_best_weights=True)  # EARLY STOPPING

    tensorboard = TensorBoard(log_dir=args.log_dir, update_freq='epoch', write_images=False,
                              write_graph=True)  # CONTROL THE TRAINING
    checkPoints = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_feq=10)
    # Train it by providing training images
    train_generator, validation_generator = train_val_images()
    train_generator, _ = train_val_images()
    #test_generator = test_images()
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    #STEP_SIZE_VALIDATION = validation_generator.n // validation_generator.batch_size
    # STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

    history = autoencoder.fit_generator(generator=train_generator,
                                         steps_per_epoch=STEP_SIZE_TRAIN,
                                         epochs=args.epochs,
                                         callbacks=[tensorboard, earlyStopping,checkPoints]
                                         #validation_data=validation_generator,
                                         #validation_steps=STEP_SIZE_VALIDATION
                                         )

    #autoencoder.fit(data_train, data_train,
    #                epochs=args.epochs,
    #                batch_size=args.batchSize,
    #                shuffle=True,
    #                validation_split=0.1,
    #                callbacks=[tensorboard,earlyStopping,checkPoints])


    os.makedirs(save_folder_path, exist_ok=True)
    with open(model_save_path, 'w+') as save_file:
        save_file.write(autoencoder.to_json())
    autoencoder.save_weights(weight_save_path)
    print("Saved model")

    # autoencoder.evaluate(data_test, data_test, batch_size=args.batchSize)
