"""
    author: Ricardo Kleinlein && Miguel Taibo
    date: 02/2020

    Script to define the Arguments class. Every script will have its own
    set of arguments as a rule, though some may be shared between tasks.
    These objects are not thought to be used independently, but simply
    as a method to automate the argument passing between scripts in the
    retrieval pipeline.
"""

import os
import argparse
import __main__ as main

class BaseArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=__doc__)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--output-dir",
            type=str,
            default="results",
            help="Directory to export the script s output to")
        self.parser.add_argument(
            "--quiet",
            action='store_true',
            help='Fewer information displayed on screen')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.args = self.parser.parse_args()
        self._correct()

        if not self.args.quiet:
            print('-' * 10 + ' Arguments ' + '-' * 10)
            print('>>> Script: %s' % (os.path.basename(main.__file__)))
            print_args = vars(self.args)
            for key, val in sorted(print_args.items()):
                print('%s: %s' % (str(key), str(val)))
            print('-' * 30)
        return self.args

    def _correct(self):
        """Assert ranges of params, mistypes..."""
        raise NotImplementedError

class CreateModelArgs(BaseArguments):
    def initialize(self):
        BaseArguments.initialize(self)

        self.parser.add_argument(
            "--longSize",
            type=int,
            default=28,
            help="Altura y anchura de la imagen que contiene la cara")
        self.parser.add_argument(
            "--downsample",
            type=int,
            default=2,
            help="Anchura de la imagen que contiene la cara")
        self.parser.add_argument(
            "--dataroot",
            type=str,
            default='/mnt/pgth04b/Data_Miguel/AutoEncoders/datasets/AFFECTNET',
            help="Carpeta de datos")
        self.parser.add_argument(
            "--log_dir",
            type=str,
            default='~/AutoEncoders/log_dir',
            help="Carpeta de Tensorboard")
        self.parser.add_argument(
            "--save_dir",
            type=str,
            default='~/AutoEncoders/data/models',
            help="Carpeta de guardado")
        self.parser.add_argument(
            "--modelname",
            type=str,
            default='autoencoder_emocional',
            help="Nombre del modelo")
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=50,
            help="# of epochs")
        self.parser.add_argument(
            "--batchSize",
            type=int,
            default=128,
            help="batch size? ")

    def _correct(self):
        assert isinstance(self.args.longSize, int)
        assert isinstance(self.args.downsample, int)
        assert isinstance(self.args.dataroot, str)
        assert isinstance(self.args.modelname, str)
        assert isinstance(self.args.epochs, int)
        assert isinstance(self.args.batchSize, int)

class FilterAffectnet_disabled(BaseArguments):
    def initialize(self):
        BaseArguments.initialize(self)

        self.parser.add_argument(
            "--th",
            type=float,
            default=0.8,
            help="Minima confianza necesaria para considerar que es una cara")
        self.parser.add_argument(
            "--dataroot",
            type=str,
            default='/mnt/pgth04b/Data_Miguel/AutoEncoders/datasets/AFFECTNET',
            help="Carpeta de datos")

    def _correct(self):
        assert isinstance(self.args.th, float)
        assert isinstance(self.args.dataroot, str)

class FilterAffectnet(BaseArguments):
    def initialize(self):
        BaseArguments.initialize(self)

        self.parser.add_argument(
            "--datapath",
            type=str,
            default='/mnt/pgth04b/Data_Miguel/AutoEncoders/datasets/AFFECTNET',
            help="Carpeta de datos")
        self.parser.add_argument(
            "--longSize",
            type=int,
            default=28,
            help="Altura y anchura de la imagen que contiene la cara")
        self.parser.add_argument(
            "--face-size",
            type=int,
            default=0,
            help="Min size (in pixel area) to keep a face at detection time [default: 0]")
        self.parser.add_argument(
            "--save-bb",
            action='store_true',
            help="Saves in memory the bounding boxes")


    def _correct(self):
        assert isinstance(self.args.datapath, str)
        assert isinstance(self.args.longSize, int)
        assert isinstance(self.args.face-size, int)

class FaceDetEncArgs(BaseArguments):
    def initialize(self):
        BaseArguments.initialize(self)

        self.parser.add_argument(
            "video_dir",
            type=str,
            default=None,
            help="Path to the directory of frames of a video")
        self.parser.add_argument(
            "--longSize",
            type=int,
            default=28,
            help="Altura y anchura de la imagen que contiene la cara")
        self.parser.add_argument(
            "--encoding_model",
            type=str,
            default='/home/migueltaibo/AutoEncoders/data/models/autoencoder_emocional_estatico/autoencoder_emocional_estatico_autoencoder_model.json',
            help="Face encoding model [default: Keras Facenet")
        self.parser.add_argument(
            "--face-size",
            type=int,
            default=0,
            help="Min size (in pixel area) to keep a face at detection time [default: 0]")
        self.parser.add_argument(
            "--save-bb",
            action='store_true',
            help="Saves in memory the bounding boxes")
        self.parser.add_argument(
            "--encoding_weights",
            type=str,
            default='/home/migueltaibo/AutoEncoders/data/models/autoencoder_emocional_estatico/autoencoder_emocional_estatico_autoencoder_weight.h5',
            help="Face encoding model [default: Keras Facenet")

    def _correct(self):
        assert os.path.isdir(self.args.video_dir)
        self.args.output_dir = os.path.dirname(
            os.path.dirname(self.args.video_dir))

class SplitFramesArgs(BaseArguments):
    def initialize(self):
        BaseArguments.initialize(self)

        self.parser.add_argument(
            "video_path",
            type=str,
            default=None,
            help="Path to a video file")
        self.parser.add_argument(
            "--fps",
            type=int,
            default=1,
            help="Frames per second")
        self.parser.add_argument(
            "--frame_height",
            type=int,
            default=854,
            help="Height of the frames")
        self.parser.add_argument(
            "--frame_width",
            type=int,
            default=480,
            help="Width of the frames")

    def _correct(self):
        assert os.path.isfile(self.args.video_path)
        assert isinstance(self.args.fps, int)
        assert isinstance(self.args.frame_height, int)
        assert isinstance(self.args.frame_width, int)

class CheckModelArgs(BaseArguments):
    def initialize(self):
        BaseArguments.initialize(self)

        self.parser.add_argument(
            "--longSize",
            type=int,
            default=28,
            help="Altura y anchura de la imagen que contiene la cara")
        self.parser.add_argument(
            "--dataroot",
            type=str,
            default='/mnt/pgth04b/Data_Miguel/AutoEncoders/datasets/AFFECTNET',
            help="Carpeta de datos")
        self.parser.add_argument(
            "--modelname",
            type=str,
            default='autoencoder_emocional',
            help="Nombre del modelo")
        self.parser.add_argument(
            "--num_it",
            type=int,
            default=10,
            help="# of images to show")

    def _correct(self):
        assert isinstance(self.args.longSize, int)
        assert isinstance(self.args.dataroot, str)
        assert isinstance(self.args.modelname, str)
        assert isinstance(self.args.num_it, int)

class DividirImagesArgs(BaseArguments):
    def initialize(self):
        BaseArguments.initialize(self)

        self.parser.add_argument(
            "--size",
            type=int,
            default=1000,
            help="Altura y anchura de la imagen que contiene la cara")
        self.parser.add_argument(
            "--dataroot",
            type=str,
            default='/mnt/pgth04b/Data_Miguel/AutoEncoders/datasets/AFFECTNET',
            help="Carpeta de datos")

    def _correct(self):
        assert isinstance(self.args.size, int)
        assert isinstance(self.args.dataroot, str)

class DbscanArgs(BaseArguments):
	def initialize(self):
		BaseArguments.initialize(self)

		self.parser.add_argument(
			"program_csv",
			type=str,
			default=None,
			help="Path to the summary of the face detection")
		self.parser.add_argument(
			"--min-part",
			type=int,
			default=120,
			help="Min time (s) to consider someone a relevant participant [default: 120]")
		self.parser.add_argument(
			"--eps_low",
			type=float,
			default=0.5,
			help="Lower Epsilon value [default: 0.5]")
		self.parser.add_argument(
			"--eps_upper",
			type=float,
			default=12.,
			help="UpperEpsilon value [default: 12]")
		self.parser.add_argument(
			"--trials",
			type=int,
			default=24,
			help="Number of experiments along the Eps axis")
		self.parser.add_argument(
			"--min_samples",
			type=int,
			default=10,
			help="Min-samples value [default: 10]")
		self.parser.add_argument(
			"--metric",
			type=str,
			default='euclidean',
			help="Distance metric in DBSCAN")
		self.parser.add_argument(
			"--min-area",
			type=int,
			default=0,
			help="Min area of face bounding box so the face is kept [default: 15000]")
		self.parser.add_argument(
			"--nthneigh",
			type=int,
			default=2,
			help="N-th neighbors (including oneself) to compute distance to [default: 2]")

	def _correct(self):
		assert os.path.isfile(self.args.program_csv)
		self.args.output_dir = os.path.join(
			os.path.dirname(self.args.program_csv), 'dbscan')
		assert isinstance(self.args.eps_low, float) or isinstance(self.args.eps_low, int)
		assert isinstance(self.args.eps_upper, float) or isinstance(self.args.eps_upper, int)
		assert isinstance(self.args.min_samples, int)
		assert isinstance(self.args.metric, str)
		dists = ['euclidean', 'cosine', 'manhattan', 'l1', 'l2']
		if self.args.metric not in dists:
			raise NotImplementedError('Pairwise distance not implemented')
		if self.args.min_area < 0:
			self.args.min_area = 0