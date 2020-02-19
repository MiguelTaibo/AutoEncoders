"""
        Filter AffectNet

        Our approach is based on visual information.
        Check every image on a dataset to get faces
        using MTCNN.

        author: Miguel Taibo
        date: 02/2020

        Usage:
            Du no yet

        options:

"""

import os

import PIL
import numpy as np
import glob
import cv2
import csv

from arguments import FilterAffectnet
from mtcnn.mtcnn import MTCNN

def divideImages(dataroot,threshold=0.8):

    face_dic_list, noface_dic_list = decideFaceNonFace(dataroot,threshold=threshold)



    n = dataroot.rfind('/')
    face_path = dataroot[0:n] + '/faces_'+dataroot[n+1:len(dataroot)]
    noface_path = dataroot[0:n] + '/nofaces_' + dataroot[n + 1:len(dataroot)]

    with open(dataroot + '/faces.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['filename', 'box', 'confidence'])
        writer.writeheader()
        for row in face_dic_list:
            try:
                filename = row['filename']
                new_filename = face_path + filename[len(dataroot):len(filename)]
                os.system('mv '+ filename + ' ' + new_filename)
                writer.writerow(row)
            except:
                continue
    with open(dataroot + '/nofaces.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['filename', 'box', 'confidence'])
        writer.writeheader()
        for row in noface_dic_list:
            try:
                filename = row['filename']
                new_filename = noface_path + filename[len(dataroot):len(filename)]
                os.system('mv '+ filename + ' ' + new_filename)
                writer.writerow(row)
            except:
                continue
    return

def getFacesDictionary(dataroot):
    detector = MTCNN()
    dic_list = []
    for filename in sorted(glob.glob(dataroot+'/*/*')):
        img = cv2.imread(filename)
        faces = detector.detect_faces(img)
        for face in faces:
            row = {
                'filename': filename,
                'box': face['box'],
                'confidence': face['confidence']
            }
            dic_list.append(row)
    return dic_list

def decideFaceNonFace(dataroot, threshold=0.8):
    ## Detectamos en todas las fotos caras
    ## Si hay caras guardamos las fotos en fac_dic_list y
    ## Si no hay guardamos las fotos en noface_dic_list
    ## En forma de diccionario
    detector = MTCNN()
    face_dic_list = []
    noface_dic_list = []
    for filename in sorted(glob.glob(dataroot + '/*/*')):
        img = cv2.imread(filename)
        faces = detector.detect_faces(img)
        th = threshold
        face_bool = False
        if len(faces) == 0:
            row = {'filename': filename, 'box': -1, 'confidence': -1}
        for face in faces:
            if face['confidence'] > th:
                th = face['confidence']
                face_bool = True
                row = {
                    'filename': filename,
                    'box': face['box'],
                    'confidence': face['confidence']
                }
            elif not face_bool:
                row = {'filename': filename, 'box': -1, 'confidence': -1}

        if face_bool:
            face_dic_list.append(row)
        else:
            noface_dic_list.append(row)

    return face_dic_list, noface_dic_list

def formatDirectories(dataroot):
    ##Create new directories to save faces and non faces
    n = dataroot.rfind('/')
    face_path = dataroot[0:n] + '/faces_'+dataroot[n+1:len(dataroot)]
    noface_path = dataroot[0:n] + '/nofaces_' + dataroot[n + 1:len(dataroot)]
    os.makedirs(face_path, exist_ok=True)
    os.makedirs(noface_path, exist_ok=True)
    ##For every directory in dataroot we create a new one in both
    ## face and noface directories
    for dirname in glob.glob(dataroot+'/*'):
        new_path = face_path + dirname[len(dataroot):len(dirname)]
        os.makedirs(new_path, exist_ok=True)
        new_path = noface_path + dirname[len(dataroot):len(dirname)]
        os.makedirs(new_path, exist_ok=True)

if __name__ == "__main__":
    args = FilterAffectnet().parse()
    formatDirectories(args.dataroot)
    divideImages(args.dataroot, threshold=args.th)