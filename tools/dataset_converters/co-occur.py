# -*- encoding: utf-8 -*-
'''
Filename         :co-occur.py
Description      :tools for the GCN based object detection
Time             :2021/12/26 12:36:46
Author           :***
Version          :1.0
'''
import json
import pickle
import itertools
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
# set global font 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


from matplotlib.ticker import MultipleLocator
from collections import Counter

import torch
from pycocotools.coco import COCO

from transformers import pipeline
from transformers import BertTokenizer, BertModel

from mmdet.core import bbox_overlaps
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(suppress=True)

def get_normal_class(cls, fault_map):
    for normal,faults in fault_map.items():
        if cls in faults:
            return normal

def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    dist = 1 - res
    # return 0.5 + 0.5 * res
    return dist

def get_spatial_feat(ann, coco):
    """ 
    input: coco annotation, [instance]
           coco instance
    return: [x1/W,y1/H,x2/W,y2/H,s/S,w/h], s is the area
    """

    imgId = ann['image_id']
    imgInfo = coco.loadImgs(imgId)[0]
    im_w = imgInfo['width']
    im_h = imgInfo['height']
    bbox = ann['bbox'] # 'bbox': [x,y,width,height]
    x,y,w,h = bbox[0],bbox[1],bbox[2],bbox[3]

    roi_area = w*h
    img_area = im_w*im_h

    spatial_feat = np.array([x/im_w, y/im_h, (x+w)/im_w, (y+h)/im_h, roi_area/img_area,w/h])

    return spatial_feat

def bbox_xywh_to_xyxy(bbox):
    """Convert bbox coordinates from (x1, y1, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [x1, y1, x1 + w, y1 + h]
    return torch.cat(bbox_new, dim=-1)

def plot_adjacency_matrix(adj_matrix,
                        labels,
                        prefix='cooccur',
                        save_dir=None,
                        title='Adjacency Matrix',
                        show_ticks=False,
                        color_theme='YlGnBu'):
    """Draw adjacency matrix with matplotlib.

    Args:
        adj_matrix (ndarray): The adjacency_matrix.
        labels (list[str]): List of class names.
    """
    # normalize the confusion matrix
    # per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    # confusion_matrix = \
    #     confusion_matrix.astype(np.float32) / per_label_sums * 100

    font = {
            'size'   : 12,
            }
    label_font = {'size': 12}
    # title_font = {'weight': 'bold', 'size': 12}

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(0.4 * num_classes, 0.4 * num_classes * 0.8), dpi=200)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(adj_matrix, cmap=cmap)
    # cbar = plt.colorbar(mappable=im, ax=ax)
    cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel('Co-occur probability', rotation=-90, va="bottom")
    cbar.ax.set_ylabel('Co-occur probability', fontdict=font)

    # ax.set_title(title, fontdict=title_font)

    plt.ylabel('Observed class', fontdict=label_font)
    plt.xlabel('Co-occur class', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    if show_ticks:
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # draw confution matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            value = adj_matrix[i, j]
            textcolor = 'white' if value > 0.5 else 'black'
            ax.text(
                j,
                i,
                '{:.2f}'.format(value),
                ha='center',
                va='center',
                color=textcolor,
                size=7)

    ax.set_ylim(len(adj_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(osp.join(save_dir, f'{prefix}_matrix.png'), format='png')


def gen_adjacency_matrix(ann_path, classes, classes_en, coco_cat_id, 
                        data_root='data/', plot_show_ticks=False):
    """Generate adjacency matrix and useful infomation.

    Args:
        classes(list): class name in coco dataset. e.g., gt.
        classes_en(list): semantic class name. e.g., tower.
        coco_cat_id(list): id number of each class.
    Note:
        1. catId is class id in coco dataset
        2. cls_index is class index in classes list
    """
    iou_mat = np.zeros((len(classes), len(classes)))
    giou_mat = np.zeros((len(classes), len(classes)))
    iof_mat = np.zeros((len(classes), len(classes)))
    cooccur_mat_obj_lvl = np.zeros((len(classes), len(classes)))
    cooccur_mat_img_lvl = np.zeros((len(classes), len(classes)))
    cooccur_mat_iof = np.zeros((len(classes), len(classes)))
    cooccur_mat_giou = np.zeros((len(classes), len(classes)))

    coco = COCO(ann_path) 
    imgIds = coco.getImgIds(catIds=[])
    tot_img_num = len(imgIds)
    tot_obj_num = len(coco.getAnnIds(imgIds=[], iscrowd=None))
    # for each img, collect its all anns
    for imgId in imgIds:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
        anns = coco.loadAnns(annIds)

        if len(anns) == 0:
            tot_img_num -= 1
            continue 

        objs_per_img = []
        # for each obj, collect (cls,x1,y1,x2,y2)
        for ann in anns:
            catId = ann['category_id']
            cls = coco.loadCats(catId)[0]['name']
            cls_index = classes.index(cls)
            pts = ann['bbox'] # [x1,y1,w,h]
            objs_per_img.append([cls_index,pts[0],pts[1],pts[2],pts[3]])

        objs_per_img = np.array(objs_per_img)
        objs_per_img = torch.from_numpy(objs_per_img)
        objs_per_img[:,1:] = bbox_xywh_to_xyxy(objs_per_img[:,1:]) # [x1,y1,x2,y2]

        ious = bbox_overlaps(objs_per_img[:,1:], objs_per_img[:,1:], 'iou')
        gious = bbox_overlaps(objs_per_img[:,1:], objs_per_img[:,1:], 'giou')
        iofs = bbox_overlaps(objs_per_img[:,1:], objs_per_img[:,1:], 'iof')

        obj_clsId_indexes = objs_per_img[:,0].int()

        # image level co-occur
        obj_clsId_indexes = obj_clsId_indexes.numpy().tolist()
        obj_cls_count = dict(Counter(obj_clsId_indexes))
        for cls_index, occur_num in obj_cls_count.items():
            cooccur_mat_obj_lvl[cls_index,cls_index] += occur_num
            cooccur_mat_img_lvl[cls_index,cls_index] += 1

        for cls_index_pair in itertools.combinations(obj_cls_count.keys(), 2):
            i,j = cls_index_pair
            cooccur_mat_obj_lvl[i,j] += obj_cls_count[j]
            cooccur_mat_obj_lvl[j,i] += obj_cls_count[i]
            cooccur_mat_img_lvl[i,j] += 1
            cooccur_mat_img_lvl[j,i] += 1

        # loop object i and object j
        for iou_index_i, cls_index_i in enumerate(obj_clsId_indexes):
            for iou_index_j, cls_index_j in enumerate(obj_clsId_indexes):
                if iou_index_i == iou_index_j: continue # filter same obj
                if cls_index_i == cls_index_j: continue
                giou_mat[cls_index_i,cls_index_j] += gious[iou_index_i,iou_index_j]
                cooccur_mat_giou[cls_index_i,cls_index_j] += 1 

                if iofs[iou_index_i,iou_index_j] != 0:
                    iou_mat[cls_index_i,cls_index_j] += ious[iou_index_i,iou_index_j]
                    iof_mat[cls_index_i,cls_index_j] += iofs[iou_index_i,iou_index_j]
                    cooccur_mat_iof[cls_index_i,cls_index_j] += 1

    cooccur_mat_iof[np.where(cooccur_mat_iof==0)] = 1 
    iou_mat = iou_mat / cooccur_mat_iof
    iof_mat = iof_mat / cooccur_mat_iof
    
    cooccur_mat_giou[np.where(cooccur_mat_giou==0)] = 1 
    giou_mat = giou_mat / cooccur_mat_giou
    giou_mat = giou_mat + 1
    giou_mat[np.where(giou_mat==1)] = 0

    # Obtain the diagonal: objects of one class in all images
    # Note: unable for COCO because of memory limit!
    # for i, cls in enumerate(classes):
    #     # collet all anns per class
    #     catId = coco.getCatIds(cls)
    #     annIds = coco.getAnnIds(imgIds=[], catIds=catId, iscrowd=None)
    #     anns = coco.loadAnns(annIds)

    #     pts = []
    #     for ann in anns:
    #         pts.append(ann['bbox']) # [r,x1,y1,w,h]

    #     pts = np.array(pts)
    #     pts = torch.from_numpy(pts)
    #     pts = bbox_xywh_to_xyxy(pts)
        
    #     ious = bbox_overlaps(pts, pts, 'giou')
    #     self_iou = torch.mean(ious) + 1 # [1]
    #     iou_mat[i,i] = self_iou
    #     iof_mat[i,i] = self_iou
    #     giou_mat[i,i] = self_iou

    # A_img_lvl
    _nums = np.diagonal(cooccur_mat_img_lvl)
    diag = _nums / tot_img_num
    A_img_lvl = cooccur_mat_img_lvl / np.max(cooccur_mat_img_lvl,axis=1).reshape(-1,1)
    A_img_lvl_eye = A_img_lvl.copy()
    A_img_lvl[np.diag_indices_from(A_img_lvl)] = diag
    A_img_lvl[A_img_lvl < 0.05] = 0.0001
    A_img_lvl_eye[A_img_lvl_eye < 0.05] = 0.0001

    # A_obj_lvl
    A_obj_lvl = cooccur_mat_obj_lvl / np.max(cooccur_mat_obj_lvl,axis=1).reshape(-1,1)
    A_obj_lvl[np.diag_indices_from(A_obj_lvl)] = diag
    A_obj_lvl[A_obj_lvl < 0.05] = 0.0001

    iou_mat[iou_mat < 0.01] = 0.0001
    iof_mat[iof_mat < 0.05] = 0.0001
    giou_mat[giou_mat < 0.05] = 0.0001

    plot_adjacency_matrix(A_img_lvl_eye, classes_en, 'A_img_lvl_eye', data_root, show_ticks=plot_show_ticks)
    plot_adjacency_matrix(A_img_lvl, classes_en, 'A_img_lvl', data_root)
    plot_adjacency_matrix(A_obj_lvl, classes_en, 'A_obj_lvl', data_root)
    plot_adjacency_matrix(iou_mat, classes_en, 'A_iou', data_root)
    plot_adjacency_matrix(iof_mat, classes_en, 'A_iof', data_root)
    plot_adjacency_matrix(giou_mat, classes_en, 'A_giou', data_root)

    iou_mat_eye = iou_mat.copy()
    iof_mat_eye = iof_mat.copy()
    giou_mat_eye = giou_mat.copy()
    iou_mat_eye[np.diag_indices_from(iou_mat_eye)] = np.ones(len(classes))
    iof_mat_eye[np.diag_indices_from(iof_mat_eye)] = np.ones(len(classes))
    giou_mat_eye[np.diag_indices_from(giou_mat_eye)] = np.ones(len(classes))

    plot_adjacency_matrix(iou_mat_eye, classes_en, 'A_iou_eye', data_root)
    plot_adjacency_matrix(iof_mat_eye, classes_en, 'A_iof_eye', data_root)
    plot_adjacency_matrix(giou_mat_eye, classes_en, 'A_giou_eye', data_root)


    adjacency_info = {'classes':classes,'coco_cat_id':coco_cat_id,
                    'A_img_lvl_eye':A_img_lvl_eye, 'A_img_lvl':A_img_lvl, 
                    'A_obj_lvl':A_obj_lvl,
                    'A_iou':iou_mat, 'A_iof':iof_mat, 'A_giou':giou_mat,
                    'A_iou_eye':iou_mat_eye, 'A_iof_eye':iof_mat_eye, 'A_giou_eye':giou_mat_eye,
                    }

    save_path = osp.join(data_root, 'adjacency_info.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(adjacency_info, file)
    print(f'save adjacency information to:{save_path}')

    return adjacency_info


def gen_cls_word_embedding(labels,
                          words=[],
                          sentences=[],
                          coco_cat_id=[],
                          data_root='data/'):
    """
    generate word embeddings of each class name with dim (1, 768)
    using the BERT pre-trianed model from https://huggingface.co/bert-base-uncased
    using dataset wikipedia and bookcorpus

    Input:
        labels: [list], annotated label names of categories. e.g., gt
        words: semantic words corresponding to classes.
            e.g., tower.
        sentences: semantic sentences corresponding to classes.
            e.g., tower is in the middle of the whole image.

    """
    
    emb_label = np.zeros((len(labels), 768))
    emb_word = np.zeros((len(labels), 768))
    emb_sentence = np.zeros((len(labels), 768))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")

    for i,label in enumerate(labels):
        encoded_input = tokenizer(label, return_tensors='pt')
        output = model(**encoded_input)
        emb_label[i] = output['pooler_output'].squeeze().detach().numpy()

    if len(words)>0:
        for i,word in enumerate(words):
            encoded_input = tokenizer(word, return_tensors='pt')
            output = model(**encoded_input)
            emb_word[i] = output['pooler_output'].squeeze().detach().numpy()

    if len(sentences)>0:
        for i,word in enumerate(words):
            encoded_input = tokenizer(word, return_tensors='pt')
            output = model(**encoded_input)
            emb_word[i] = output['pooler_output'].squeeze().detach().numpy()

    print('word embeddings size:', emb_label.shape)

    embedding_dict = {
        'coco_cat_id':coco_cat_id,
        'labels':labels,
        'words':words,
        'sentences':sentences,
        'emb_label':emb_label, 
        'emb_word':emb_word,
        'emb_sentence':emb_sentence,
        'embeddings':emb_label,}

    save_path = osp.join(data_root, 'cls_embedding.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(embedding_dict, file)

    print(f'save embeddings to:{save_path}')       

    return embedding_dict

# voc 
classes_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
coco_cat_id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
classes_words = classes_labels
classes_sentances = []
data_root = 'data/VOCdevkit'
ann_path = 'data/VOCdevkit/voc0712_trainval.json'


# coco 
# classes_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
# 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
# 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
# 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
# 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
# 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
# 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
# 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
# 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
# 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
# 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
# 'teddy bear', 'hair drier', 'toothbrush']
# classes_words = classes_labels
# classes_sentances = []
# coco_cat_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
# 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
# 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 
# 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
# data_root = 'data/coco'
# ann_path = 'data/coco/annotations/instances_train2017.json'

# check the class name and id
cls_names = []
cls_ids = []
with open(ann_path, 'r') as fp:
  coco_ann = json.load(fp)  # 加载json文件
for item in coco_ann['categories']:
    cls_names.append(item['name'])
    cls_ids.append(item['id'])

print('cls_names:', cls_names)
print('cls_ids:', cls_ids)

assert len(list(classes_labels))==len(coco_cat_id),'class length not fit: prepare({a}) - load({b})'.format(a=len(list(classes_labels)),b=len(cls_names))
diff = set(classes_labels).difference(set(cls_names))
assert classes_labels==cls_names, f'class name not fit:{diff}'
assert coco_cat_id==cls_ids, 'class id not fit'

print('Start generate co-occur information......')


adjacency_info = gen_adjacency_matrix(
    ann_path, 
    classes_labels,
    classes_words,
    coco_cat_id,
    data_root,
    plot_show_ticks=True)

cls_embedding = gen_cls_word_embedding(
    classes_labels,
    classes_words, 
    classes_sentances,
    coco_cat_id,
    data_root=data_root)
