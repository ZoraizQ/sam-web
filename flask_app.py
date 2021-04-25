import os
CODE_DIR = "./SAM"
os.chdir(CODE_DIR)

from PIL import Image
import numpy as np
from argparse import Namespace
import sys
import pprint
import torch
import torchvision.transforms as transforms
import flask
import flask_cors
from flask import Flask, redirect, url_for, request
from flask_cors import CORS,cross_origin
from flask_ngrok import run_with_ngrok
import json


sys.path.append(".")
sys.path.append("..")

from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp


EXPERIMENT_TYPE = 'ffhq_aging'
EXPERIMENT_DATA_ARGS = {
    "ffhq_aging": {
        "model_path": "../../sam_ffhq_aging.pt",
        "image_path": "notebooks/images/866.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]
model_path = EXPERIMENT_ARGS['model_path']

ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
pprint.pprint(opts)
# update the training options
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

def run_alignment(image_path):
    import dlib
    from scripts.align_all_parallel import align_face
    predictor = dlib.shape_predictor("../../shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 


def run_on_batch(inputs, net):
    result_batch = np.zeros(inputs.shape)
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch


i = 0
app = Flask(__name__)
run_with_ngrok(app)
CORS(app, support_credentials=True)

@app.route('/predict', methods=['POST'])

def predict():
    global i

    if request.method =='POST' and request.files.getlist('media'):
        up_file_list = []

        fpath = None

        for a_file in request.files.getlist('media'):
            print(a_file)
            if a_file.filename:
                fpath = 'i'+str(i)+'_'+a_file.filename
                # Save file 
                a_file.save(fpath)
                up_file_list.append(fpath)

        aligned_image = run_alignment(fpath)
        aligned_image.resize((256, 256))
        img_transforms = EXPERIMENT_ARGS['transform']
        input_image = img_transforms(aligned_image)

        target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        age_transformers = [AgeTransformer(target_age=age) for age in target_ages]

        results = np.array(aligned_image.resize((1024, 1024)))

        with torch.no_grad():
            for age_transformer in age_transformers:
                print(f"Running on target age: {age_transformer.target_age}")
                input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
                input_image_age = torch.stack(input_image_age)
                result_tensor = run_on_batch(input_image_age, net)[0]
                result_image = tensor2im(result_tensor)
                results = np.concatenate([results, result_image], axis=1)

        pred = Image.fromarray(results)	
        pred_fname = 'p'+fpath
        pred.save(pred_fname)

        return send_file(pred_fname, mimetype='image/gif')

    return {'error': -1}

app.run()
