import random
import os
import torch
import json
import math
import ast
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPProcessor
from torchvision import transforms
import logging
logger = logging.getLogger(__name__)


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

class NonSemanticAug(object):
    """Apply non semantic augmentation to an image: randomly crop patches from the image and apply flips and rotations and 
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, scale=(16, 48), n_non_sematic=10):
        self.horflip_helper = transforms.RandomHorizontalFlip()
        self.verflip_helper = transforms.RandomVerticalFlip()
        self.rotate_helper = RandomRotation()
        self.n_non_sematic = n_non_sematic
        # self.scale_a = scale
        self.scale_a = 16
        self.scale_b = 48

    def __call__(self, img_ori):
        img_reorders = []
        for _ in range(self.n_non_sematic):
            patch_size = random.randint(self.scale_a,self.scale_b)
            n_patch = 224 // patch_size + 1
            reorder_size = patch_size*n_patch
            img_reorder = torch.zeros((3, reorder_size, reorder_size))
            for i in range(n_patch):
                for j in range(n_patch):
                    w = np.random.randint(0, 224-patch_size)
                    h = np.random.randint(0, 224-patch_size)
                    img_reorder[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = self.horflip_helper(self.verflip_helper(self.rotate_helper(img_ori[:, h:h+patch_size, w:w+patch_size])))
                    # img_reorder[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = img_ori[:, h:h+patch_size, w:w+patch_size]
            img_reorders.append(img_reorder[:, :224, :224].clone())
        return img_reorders

class DAIE_Dataset(Dataset):
    def __init__(self, data_path=None, img_path=None, clip_path=None, mode='train') -> None:
        self.data_path = data_path
        self.img_path = img_path
        self.mode = mode
        self.text_path = self.data_path[self.mode]


        with open(self.text_path) as f:
            self.dataset = json.load(f)
        # self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_path)
        self.patch_re_image = NonSemanticAug(scale=32, n_non_sematic=10)


    def image_process(self, image_path):
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
        image = Image.open(image_path)
        image = transform(image)
        return image
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # for val and test dataset, the sample[2] is hashtag label
        if self.mode == 'train':
            label = sample[2]
            text = sample[3]
        else:
            label = sample[3]
            text = sample[4]

        twitter = text["token_cap"]
        dep = text["token_dep"]

        img = os.path.join(self.img_path, sample[0]+'.jpg')
        # img = Image.open(img)
        # img = self.clip_processor(img)
        img = self.image_process(img)

        img_pri = torch.stack(self.patch_re_image(img))

        return label, twitter, dep, img, img_pri


class PadCollate:
    def __init__(self, label_dim=0, twitter_dim=1, dep_dim=2, img_dim=3, img_pri_dim=4, clip_path=None):
        """
        Args:
            img_dim (int): dimension for the image bounding boxes
            embed_dim1 (int): dimension for the matching caption
            embed_dim2 (int): dimension for the non-matching caption
            type
        """

        self.label_dim = label_dim
        self.twitter = twitter_dim
        self.dep = dep_dim
        self.img_dim = img_dim
        self.img_pri_dim = img_pri_dim
        
        self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_path)
        # self.clip_processor = CLIPProcessor.from_pretrained(clip_path)

    def pad_collate(self, batch):
        # img -> batch
        imgs_batch = list(map(lambda t: t[self.img_dim].clone().detach(), batch))
        imgs_batch = torch.stack(imgs_batch)
        # img_pri -> batch
        imgs_pri_batch = list(map(lambda t: t[self.img_pri_dim].clone().detach(), batch))
        imgs_pri_batch = torch.stack(imgs_pri_batch)
        # text -> token
        twitters = list(map(lambda t: t[self.twitter], batch))
        token_lens = [len(twitter) for twitter in twitters]
        texts_token = self.clip_tokenizer(text=twitters, is_split_into_words=True, return_tensors="pt", truncation=True,
                                     max_length=77, padding=True)
        # labels: 1 is sarcasm, 0 is non-sarcasm
        labels = torch.tensor(list(map(lambda t: t[self.label_dim], batch)), dtype=torch.long)

        # text preprocess
        word_spans = []
        word_len = []
        for index_encode, len_token in enumerate(token_lens):
            word_span_ = []
            for i in range(len_token):
                word_span = texts_token[index_encode].word_to_tokens(i)
                if word_span is not None:
                    # delete [CLS]
                    word_span_.append([word_span[0] - 1, word_span[1] - 1])
            word_spans.append(word_span_)
            word_len.append(len(word_span_))
        # text mask
        # mask矩阵是相对于word token的  key_padding_mask for computing the importance of each word in txt_encoder and
        # interaction modules
        max_word_len = max(word_len)
        word_mask = construct_mask_text(word_len, max_word_len)
        
        # img preprocess
        img_edge_index = construct_edge_image(49) # ViT's 50 = global batch + 7×7(49 batch)
        # img mask
        imgs_patch_len = [len(img_l) for img_l in imgs_batch]
        max_img_patch_len = max(imgs_patch_len)
        img_mask = construct_mask_text(imgs_patch_len, max_img_patch_len)

        # textual graph preprocess
        deps1 = [x[self.dep] for x in batch]
        deps1_ = []
        # to avoid index out of range
        for dep in deps1:
            deps1_.append([d for d in dep if d[0] < max_word_len and d[1] < max_word_len])
        org_chunk = [torch.arange(i, dtype=torch.long) for i in word_len]

        dep_se, gnn_mask, np_mask = construct_edge_text(deps=deps1_, max_length=max_word_len,
                                                               chunk=org_chunk)
        
        return imgs_batch, imgs_pri_batch, texts_token, labels,\
            word_spans, word_len, word_mask, dep_se, gnn_mask, np_mask, img_mask, img_edge_index

    def __call__(self, batch):
        return self.pad_collate(batch)


def construct_mask_text(seq_len, max_length):
    """

    Args:
        seq_len1(N): list of number of words in a caption without padding in a minibatch
        max_length: the dimension one of shape of embedding of captions of a batch

    Returns:
        mask(N,max_length): Boolean Tensor
    """
    # the realistic max length of sequence
    max_len = max(seq_len)
    if max_len <= max_length:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool), torch.ones(max_length - len, dtype=bool)]) for len in seq_len])
    else:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool),
                        torch.ones(max_length - len, dtype=bool)]) if len <= max_length else torch.zeros(max_length,
                                                                                                         dtype=bool) for
             len in seq_len])

    return mask

# construct edge [CLS]
def construct_edge_text(deps, max_length, chunk=None):
    """

    Args:
        deps: list of dependencies of all captions in a minibatch
        chunk: use to confirm where
        max_length : the max length of word(np) length in a minibatch
        use_np:

    Returns:
        deps(N,2,num_edges): list of dependencies of all captions in a minibatch. with out self loop.
        gnn_mask(N): Tensor. If True, mask.
        np_mask(N,max_length+1): Tensor. If True, mask
    """
    dep_se = []
    gnn_mask = []
    np_mask = []
    for i, dep in enumerate(deps):
        if len(dep) > 3 and len(chunk[i]) > 1:
            dep_np = [torch.tensor(dep, dtype=torch.long), torch.tensor(dep, dtype=torch.long)[:, [1, 0]]]
            gnn_mask.append(False)
            np_mask.append(True)
            dep_np = torch.cat(dep_np, dim=0).T.contiguous()
        else:
            dep_np = torch.tensor([])
            gnn_mask.append(True)
            np_mask.append(False)
        dep_se.append(dep_np.long())

    np_mask = torch.tensor(np_mask).unsqueeze(1)
    np_mask_ = [torch.tensor(
        [True] * max_length) if gnn_mask[i] else torch.tensor([True] * max_length).index_fill_(0, chunk_,
                                                                                               False).clone().detach()
                for i, chunk_ in enumerate(chunk)]
    np_mask_ = torch.stack(np_mask_)
    np_mask = torch.cat([np_mask_, np_mask], dim=1)
    gnn_mask = torch.tensor(gnn_mask)
    return dep_se, gnn_mask, np_mask

def construct_edge_image(num_patches):
    """
    Args:
        num_patches: the patches of image (49)
    There are two kinds of construct method
    Returns:
        edge_image(2,num_edges): List. num_edges = num_boxes*num_boxes
    """
    # fully connected
    edge_image = []
    # for i in range(num_patches):
    #     edge_image.append(torch.stack([torch.full([num_patches], i, dtype=torch.long),
    #                                    torch.arange(num_patches, dtype=torch.long)]))
    # edge_image = torch.cat(edge_image, dim=1)
    # remove self-loop
    p = math.sqrt(num_patches)
    for i in range(num_patches):
        for j in range(num_patches):
            if j == i:
                continue
            if math.fabs(i % p - j % p) <= 1 and math.fabs(i // p - j // p) <= 1:
                edge_image.append([i, j])
    edge_image = torch.tensor(edge_image, dtype=torch.long).T
    return edge_image