# Marginalization experiments

import os
import pdb
import json
import torch
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from model_zoo import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", default="Controlled_Images_A", type=str, choices=["Controlled_Images_A", "Controlled_Images_B", "COCO_QA_one_obj", "COCO_QA_two_obj", "VG_QA_one_obj", "VG_QA_two_obj"])
    parser.add_argument("--negative", default="COCO", type=str, choices=['COCO', 'VG']) 
    parser.add_argument("--model-name", default="openai-clip:ViT-B/32", type=str)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, image_preprocess = get_model(args.model_name, device)

    text_features_positive = {}
    image_features_positive = {}
    image_features_negative = {}

    if args.positive == 'Controlled_Images_A':
        positive_data = 'data/controlled_images_dataset.json'
    elif args.positive == 'Controlled_Images_B':
        positive_data = 'data/controlled_clevr_dataset.json'
    elif args.positive == 'COCO_QA_one_obj':
        positive_data = 'data/coco_qa_one_obj.json'
    elif args.positive == 'COCO_QA_two_obj':
        positive_data = 'data/coco_qa_two_obj.json'
    elif args.positive == 'VG_QA_one_obj':
        positive_data = 'data/vg_qa_one_obj_v2.json'
    elif args.positive == 'VG_QA_two_obj':
        positive_data = 'data/vg_qa_two_obj_v2.json'
    else:
        print("???")
        exit()

    # Read positive data, get image and text vectors for each of them.
    for d in tqdm(json.load(open(positive_data))):
        if 'COCO' in args.positive:
            filename = 'data/val2017/{}.jpg'.format(str(d[0]).zfill(12))
            text = d[1:]
        elif 'VG' in args.positive:
            filename = 'data/vg_images/{}.jpg'.format(d[0])
            text = d[1:]
        else:
            filename = d['image_path']
            text = d['caption_options']
        image = image_preprocess(Image.open(filename)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.get_image_embeddings([image], marginalization=True) 
            #model.model.encode_image(image)
            text_features = model.get_text_embeddings(text)  
            #model.model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features_positive[filename] = image_features
            text_features_positive[filename] = text_features

    
    if args.negative == 'COCO':
        negative_images = os.listdir('data/val2017')
    elif args.negative == 'VG':
        negative_images = os.listdir('data/vg_images')
    else:
        print("???")
        exit()

    # Read negative images only, get vectors for each of them.
    for i, neg_file in enumerate(tqdm(negative_images)):
        if args.negative == 'COCO':
            filename = 'data/val2017/' + neg_file
        elif args.negative == 'VG':
            filename = 'data/vg_images/'+ neg_file
        image = image_preprocess(Image.open(filename)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.get_image_embeddings([image], marginalization=True)  
            #model.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features_negative[neg_file] = image_features

    # For each positive caption, adjust its dot product based on its
    # average dot product with all the negative images.
    correct = 0
    correct_list = []
    incorrect_list = []
    diff_list = []
    for i, filename in enumerate(tqdm(image_features_positive.keys())):
        text_features = text_features_positive[filename]
        image_features = image_features_positive[filename]
        # shape [101, 512]
        all_image_features = torch.vstack([image_features] + list(image_features_negative.values()))
        avg_dot_products_with_negative = torch.mean(all_image_features[1:] @ text_features.T, dim=0)
        dot_products_with_positive = (all_image_features[:1] @ text_features.T)[0]
        differences = dot_products_with_positive - avg_dot_products_with_negative
        diff_list.append((filename.split('/')[-1].split('.')[0], \
                avg_dot_products_with_negative[0].item()))
        if torch.argmax(differences).item() == 0:
            correct += 1
            correct_list.append(filename)
        else:
            incorrect_list.append(filename)
        
    #print(correct)
    print(correct*100/len(image_features_positive))
    #print(np.mean(diff_list))
    #print(np.mean([d for d in diff_list if d>0]))
    #print(np.mean([d for d in diff_list if d<0]))
    #print(np.mean([i[-1] for i in incorrect_list]))
    #print(sum([i[-1] for i in incorrect_list])*100/len(positive_filenames))
    #pdb.set_trace()
    #print()
    #json.dump(incorrect_list, open('vg_incorrect_for_sample_{}.json'.format(SAMPLE_SIZE), 'w'))

    if args.positive == 'Controlled_Images_A':
        eval_dict = {(d.split('/')[-1].split('_')[0], \
                        d.split('/')[-1].split('_')[-1][:-5]): \
                        {'left': 0, 'right': 0, \
                        'on': 0, 'under': 0} for d in \
                        list(image_features_positive.keys())}
    elif args.positive == 'Controlled_Images_B':
        eval_dict = {(d.split('/')[-1].split('_')[0], \
                        d.split('/')[-1].split('_')[-1][:-5]): \
                        {'left': 0, 'right': 0, \
                        'in-front': 0, 'behind': 0} for d in \
                        list(image_features_positive.keys())}
    if 'Controlled' in args.positive:
        accuracy_dict = {k: 1 for k in correct_list}
        accuracy_dict.update({k: 0 for k in incorrect_list})
        for d, correct in accuracy_dict.items():
            eval_dict[(d.split('/')[-1].split('_')[0], \
                d.split('/')[-1].split('_')[-1][:-5])][d.split('/')[-1].split('_')[1]] = correct

        pair_correct = 0
        set_correct = 0
        for obj_pair, correct_dict in eval_dict.items():
            if correct_dict['left'] and correct_dict['right']:
                pair_correct += 1
            if 'A' in args.positive:
                if correct_dict['on'] and correct_dict['under']:
                    pair_correct += 1
            else:
                if correct_dict['in-front'] and correct_dict['behind']:
                    pair_correct += 1
            if sum(correct_dict.values()) == 4:
                set_correct += 1
        pair_accuracy = pair_correct*100/(len(accuracy_dict)/2)
        set_accuracy = set_correct*100/(len(accuracy_dict)/4)
        print("Pair accuracy: {}".format(pair_accuracy))
        print("Set accuracy: {}".format(set_accuracy))

    #pdb.set_trace()
    print()


if __name__ == '__main__':
    main()




