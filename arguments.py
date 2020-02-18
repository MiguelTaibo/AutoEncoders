"""
	author: Ricardo Kleinlein
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
            "--height",
            type=int,
            default=28,
            help="Altura de la imagen que contiene la cara")
        self.parser.add_argument(
            "--width",
            type=int,
            default=28,
            help="Anchura de la imagen que contiene la cara")
        self.parser.add_argument(
            "--dataroot",
            type=str,
            default='/home/migueltaibo/AutoEncoders/datasets',
            help="Carpeta de datos")

    def _correct(self):
        assert isinstance(self.args.height, int)
        assert isinstance(self.args.width, int)