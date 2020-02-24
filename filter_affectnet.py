"""
    Filter AffectNet

    Go through every image on a dataset recognising faces
    using MTCNN.
    For each face take its bounding box

    author: Ricardo Kleinlein && Miguel Taibo
    date: 02/2020

    Usage:
        python filter_affectnet.py --datapath <datapath>

    Options:
        --face-size :   minimun face size to detect
        --save-bb   :   save detected faces boundingboxes
        --ouput-dir :   directory to save results
"""


import os
import csv
from PIL import Image
import pandas as pd
import numpy as np


from mtcnn.mtcnn import MTCNN
from os.path import join
from arguments import FilterAffectnet
from tqdm import tqdm

fix_coord = lambda x: 0 if x < 0 else x

def update(program_info, frame_info, output_dir='results', save_bb=False):
    """Append the information of the current frame to the
    program overall knowledge.

    Args:
        program_info (dict, list): Paths and features of each face
        frame_info (dict, list): Vectors and features of each face
        output_dir (str, optional): Output directory
        save_bb (bool, optional): Whether or not save a copy of the bbs

        Return:
            an updated version of the program_info
    """
    init_item = len(program_info['img'])  # Global iterator idx
    if len(frame_info['img']) > 0:
        for item in range(len(frame_info['img'])):
            program_info['size'].append(frame_info['size'][item])

            program_info['confidence'].append(frame_info['confidence'][item])

            img_path = join(
                output_dir, 'boundingbox', 'img_' + str(init_item) + '.png')
            if save_bb:
                program_info['img'].append(img_path)
                img = Image.fromarray(frame_info['img'][item])
                img.save(img_path)
            else:
                program_info['img'].append('not_saved')

            # vector_path = join(
            #     output_dir, 'embedding', 'embedding_' + str(init_item))
            # np.save(vector_path, frame_info['embedding'][item], allow_pickle=True)
            # program_info['embedding'].append(vector_path + '.npy')
            program_info['framepath'].append(frame_info['framepath'][item])

            init_item += 1

    return summary


def detect(frame, framepath, size_threshold, detection_model):
    """MTCNN Face detection for an image.

    Args:
        frame (float): np.ndarray with the frame pixels
        framepath (str): Path to the frame it belongs to
        size_threshold (int): min area of face in pixels
        detection_model (keras.Model): Detection model

    Return:
        dict of lists of the face images detected
        and their sizes and confidence in detection
    """
    faces = detection_model.detect_faces(frame)
    frame_info = {'framepath': [],
                  'img': [],
                  'size': [],
                  'confidence': []}
    are_faces = True if len(faces) > 0 else False
    if are_faces:
        for face in faces:
            coord = face['box']  # [x0, y0, width, height]
            coord[0] = fix_coord(coord[0])
            coord[1] = fix_coord(coord[1])
            conf = face['confidence']
            # face_size = coord[2] * coord[3]	# area
            face_size = (2 * coord[2]) + (2 * coord[3])  # length
            if face_size >= size_threshold:

                cropped_face = frame[coord[1]:coord[1]+coord[3],coord[0]:coord[0]+coord[2]]
                cropped_face = Image.fromarray(cropped_face).resize((28, 28))
                cropped_face = np.asarray(cropped_face)

                frame_info['img'].append(cropped_face)
                frame_info['size'].append(face_size)
                frame_info['confidence'].append(conf)
                frame_info['framepath'].append(framepath)

    return frame_info


if __name__ == '__main__':
    args = FilterAffectnet().parse()

    image_list = []
    dir_list = sorted(os.listdir(args.datapath))

    for directory in tqdm(dir_list):
        im_list = sorted(os.listdir(join(args.datapath,directory)))
        for im in im_list:
            image_list.append(join(args.datapath,directory,im))

    detection_model = MTCNN()

    summary = {'framepath': [],
               'img': [],
               'size': [],
               'confidence': []}
    if args.save_bb:
        os.makedirs(join(args.output_dir, 'boundingbox'), exist_ok=True)

    for img_path in tqdm(image_list):
        image = np.asarray(Image.open(img_path).convert('RGB'))
        faces = detect(image,
                       img_path,
                       args.face_size,
                       detection_model)

        summary = update(program_info=summary,
                         frame_info=faces,
                         output_dir=args.output_dir,
                         save_bb=args.save_bb)

    pd.DataFrame(summary).to_csv(join(
        args.output_dir, 'detection_and_encoding.csv'), index=None)