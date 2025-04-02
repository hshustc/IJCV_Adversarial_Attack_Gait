# encoding: utf-8
import os
import os.path as osp
import shutil
import math
import random
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

def array2img(x):
    return (x*255.0).astype('uint8')

def img2array(x):
    return x.astype('float32')/255.0

def save_seq(seq, seq_dir):
    if osp.exists(seq_dir):
        shutil.rmtree(seq_dir)
    if not osp.exists(seq_dir):
        os.makedirs(seq_dir)
    for i in range(seq.shape[0]):
        save_name = osp.join(seq_dir, '{:0>3d}.png'.format(i))
        cv2.imwrite(save_name, array2img(seq[i, :, :]))

def merge_seq(seq, row=6, col=6):
    frames_index = np.arange(seq.shape[0])
    im_h = seq.shape[1]
    im_w = seq.shape[2]
    num_per_im = row*col
    if len(frames_index) < num_per_im:
        selected_frames_index = sorted(np.random.choice(frames_index, num_per_im, replace=True))
    else:
        selected_frames_index = sorted(np.random.choice(frames_index, num_per_im, replace=False))
    im_merged = np.zeros((im_h*row, im_w*col))
    for i in range(len(selected_frames_index)):
        im = seq[selected_frames_index[i], :, :]
        y = int(i/col)
        x = i%col
        im_merged[y*im_h:(y+1)*im_h, x*im_w:(x+1)*im_w] = im
    im_merged = array2img(im_merged)
    return im_merged

def pad_seq(seq, pad_size):
    return np.pad(seq, ([0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]]), mode='constant')

def lean_seq(seq):
    s, h, w = seq.shape
    top_left = (seq[:, 0:int(h/2), 0:int(w/2)] > 0).sum() / float(s)
    top_right = (seq[:, 0:int(h/2), int(w/2):w] > 0).sum() / float(s)
    bottom_left = (seq[:, int(h/2):h, 0:int(w/2)] > 0).sum() / float(s)
    bottom_right = (seq[:, int(h/2):h, int(w/2):w] > 0).sum() / float(s)
    if top_left > 1.5 * top_right and bottom_right > 1.5 * bottom_left:
        return 'left'
    if top_right > 1.5 * top_left and bottom_left > 1.5 * bottom_right:
        return 'right'
    return None

def cut_img(img, T_H, T_W):
    # print("before cut_img: ", img.shape, np.min(img), np.max(img), T_H, T_W, img.dtype)
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right].astype('uint8')
    # print("after cut_img: ", img.shape, np.min(img), np.max(img), T_H, T_W, img.dtype)
    return img

class BasicSilAlign(object):
    def __init__(self):
        pass
    
    def __call__(self, seq):
        _, h, w = seq.shape
        seq = array2img(seq)
        seq = [cut_img(seq[tmp, :, :], h, w) for tmp in range(seq.shape[0])]
        seq = img2array(np.stack(seq))
        return seq

class RandomErasing(object):
    def __init__(self, prob=0.5, sl=0.02, sh=0.05, r1=0.3, per_frame=False):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.per_frame = per_frame

    def __call__(self, seq):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                for attempt in range(100):
                    seq_size = seq.shape
                    area = seq_size[1] * seq_size[2]
        
                    target_area = random.uniform(self.sl, self.sh) * area
                    aspect_ratio = random.uniform(self.r1, 1 / self.r1)
        
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
        
                    if w < seq_size[2] and h < seq_size[1]:
                        x1 = random.randint(0, seq_size[1] - h)
                        y1 = random.randint(0, seq_size[2] - w)
                        seq[:, x1:x1+h, y1:y1+w] = 0.
                        return seq
            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...]) for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)

class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10, lean=False):
        self.prob = prob
        self.degree = degree
        self.lean = lean
    
    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            '''
            img_h = seq.shape[1]
            img_w = seq.shape[2]
            angle = self.get_params(seq)
            seq = array2img(seq)
            # Rotate a given image to the given number of degrees counter clockwise around its centre.
            seq = [Image.fromarray(seq[tmp, :, :], mode='L').rotate(angle) for tmp in range(seq.shape[0])]
            return img2array(np.stack(seq))
            '''
            _, dh, dw = seq.shape
            # rotation
            degree = self.get_params(seq)
            M1 = cv2.getRotationMatrix2D((dh // 2, dw // 2), degree, 1) # Angle is positive for anti-clockwise and negative for clockwise
            # affine
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            return seq
    
    def get_params(self, seq):
        if not self.lean:
            angle = random.uniform(-self.degree, self.degree)
        else:
            flag = lean_seq(seq)
            if flag == 'left':
                angle = random.uniform(-self.degree, 0)
                # angle =  -self.degree
            elif flag == 'right':
                angle = random.uniform(0, self.degree)
                # angle = self.degree
            else:
                angle = random.uniform(-self.degree, self.degree)
        return angle

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            return seq[:, :, ::-1]

class RandomPadCrop(object):
    def __init__(self, prob=0.5, pad_size=(4, 0), per_frame=False):
        self.prob = prob
        self.pad_size = pad_size
        self.per_frame = per_frame
    
    def __call__(self, seq):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                _, dh, dw = seq.shape
                seq = pad_seq(seq, self.pad_size)
                _, sh, sw = seq.shape
                bh, lw, th, rw = self.get_params((sh, sw), (dh, dw))
                seq = seq[:, bh:th, lw:rw]
                return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...]) for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, axis=0)

    def get_params(self, src_size, dst_size):
        sh, sw = src_size
        dh, dw = dst_size
        if sh == dh and sw == dw:
            return 0, 0, dh, dw

        i = random.randint(0, sh - dh)
        j = random.randint(0, sw - dw)
        return i, j, i+dh, j+dw

class RandomPerspective(object):
    def __init__(self, prob=0.5, h_range=10, w_range=10, lean=False):
        self.prob = prob
        self.h_range = h_range
        self.w_range = w_range
        self.lean = lean
    
    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            raw_seq = seq
            _, h, w = seq.shape
            TL, TR, BL, BR = self.get_params(seq)
            srcPoints = np.float32([TL, TR, BL, BR])
            canvasPoints = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            perspectiveMatrix = cv2.getPerspectiveTransform(np.array(srcPoints), np.array(canvasPoints))
            seq = [cv2.warpPerspective(_[0, ...], perspectiveMatrix, (w, h)) 
                        for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...] for _ in seq], 0)
            #############################################################
            for tmp in range(seq.shape[0]):
                tmp_sil = array2img(seq[tmp, :, :])
                if tmp_sil.sum() <= 10000: # Illegal Perspective
                    print('Illegal Perspective: srcPoints={}, canvasPoints={}'.format(srcPoints, canvasPoints))
                    return raw_seq 
            #############################################################
            return seq
    
    def get_params(self, seq):
        _, h, w = seq.shape
        '''
        x_left = list(range(-self.w_range, self.w_range + 1))
        x_right = list(range(w - self.w_range, w + self.w_range + 1))
        y_top = list(range(-self.h_range, self.h_range + 1))
        y_bottom = list(range(h - self.h_range, h + self.h_range + 1))
        '''
        x_left = list(range(-self.w_range, self.w_range + 1))
        x_right = list(range(w - self.w_range, w + self.w_range + 1))
        y_top = [0]
        y_bottom = [h]
        if not self.lean:
            TL = (random.choice(x_left), random.choice(y_top))
            TR = (random.choice(x_right), random.choice(y_top))
            BL = (random.choice(x_left), random.choice(y_bottom))
            BR = (random.choice(x_right), random.choice(y_bottom))
        else:
            x_left_neg = list(range(-self.w_range, 0 + 1))
            x_left_pos = list(range(0, self.w_range + 1))
            x_right_neg = list(range(w - self.w_range, w + 1))
            x_right_pos = list(range(w, w + self.w_range + 1))
            # x_left_neg = [-self.w_range]
            # x_left_pos = [self.w_range]
            # x_right_neg = [w - self.w_range]
            # x_right_pos = [w + self.w_range]
            flag = lean_seq(seq)
            if flag == 'left':
                TL = (random.choice(x_left_neg), random.choice(y_top))
                TR = (random.choice(x_right_neg), random.choice(y_top))
                BL = (random.choice(x_left_pos), random.choice(y_bottom))
                BR = (random.choice(x_right_pos), random.choice(y_bottom))
            elif flag == 'right':
                TL = (random.choice(x_left_pos), random.choice(y_top))
                TR = (random.choice(x_right_pos), random.choice(y_top))
                BL = (random.choice(x_left_neg), random.choice(y_bottom))
                BR = (random.choice(x_right_neg), random.choice(y_bottom))
            else:
                TL = (random.choice(x_left), random.choice(y_top))
                TR = (random.choice(x_right), random.choice(y_top))
                BL = (random.choice(x_left), random.choice(y_bottom))
                BR = (random.choice(x_right), random.choice(y_bottom))
        return TL, TR, BL, BR

def build_data_transforms(augment_config, resolution=64, random_seed=2019):
    # config: ['erasing_0.5', 'flip_0.5', 'rotate_0.5', 'persp_0.5', 'crop_0.5']
    print("augment_config={}, random_seed={} for build_data_transforms".format(augment_config, random_seed))
    np.random.seed(random_seed)
    random.seed(random_seed)

    augment_dict = {}
    for aug in augment_config:
        aug_type, aug_prob = aug.split('_')
        assert(aug_type in ['erase', 'rotate', 'flip', 'crop', 'persp'])
        assert(float(aug_prob) >= 0.0 and float(aug_prob) <= 1.0)
        augment_dict.update({aug_type:float(aug_prob)})
    
    object_list = []
    if 'persp' in augment_dict.keys():
        object_list.append(RandomPerspective(prob=augment_dict['persp'], h_range=10, w_range=10, lean=True))
    if 'rotate' in augment_dict.keys():
        object_list.append(RandomRotate(prob=augment_dict['rotate'], degree=20, lean=True))
    if 'crop' in augment_dict.keys():
        object_list.append(RandomPadCrop(prob=augment_dict['crop'], pad_size=(4*int(resolution/64), 0), per_frame=True))
    object_list.append(BasicSilAlign())
    if 'flip' in augment_dict.keys():
        object_list.append(RandomHorizontalFlip(prob=augment_dict['flip']))
    if 'erase' in augment_dict.keys():
        object_list.append(RandomErasing(prob=augment_dict['erase'], sl=0.02, sh=0.05, r1=0.3, per_frame=False))
    transform = T.Compose(object_list)
    return transform

def build_seq_list(root_dir):
    all_seq_list = []
    for root, dirs, files in os.walk(root_dir):
        for _file in files:
            if _file.endswith('.pkl'):
                seq_path = osp.join(root, _file)
                all_seq_list.append(seq_path)
                print("{} is INCLUDED".format(seq_path))
    return all_seq_list

if __name__ == "__main__":
    import pickle
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    SEED = 2020
    np.random.seed(SEED)
    random.seed(SEED)
    
    src_dir = './example_data'
    des_dir = './augment_data'
    all_seq_list = build_seq_list(src_dir)

    for seq_path in all_seq_list:
        seq_dir = seq_path[:-len(osp.basename(seq_path))]
        save_dir = seq_dir.replace(src_dir, des_dir)
        merge_imgs = {}

        seq_in = pickle.load(open(seq_path, 'rb'))
        resolution = seq_in.shape[1]
        cut_padding = 10*int(resolution/64)
        seq_in = seq_in[:, :, cut_padding:-cut_padding]
        seq_in = img2array(seq_in)
        save_seq(seq_in, seq_dir=osp.join(save_dir, 'raw_seq'))
        merge_imgs.update({'raw':merge_seq(seq_in)})
        print(seq_in.shape, np.min(seq_in), np.max(seq_in), seq_in.dtype)

        transform = build_data_transforms(['rotate_1.0'])
        seq_out = transform(seq_in.copy())
        save_seq(seq_out, seq_dir=osp.join(save_dir, 'rotate_seq'))
        seq_merge = merge_seq(seq_out)
        merge_imgs.update({'rotate':merge_seq(seq_out)})
        print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

        transform = build_data_transforms(['persp_1.0'])
        seq_out = transform(seq_in.copy())
        save_seq(seq_out, seq_dir=osp.join(save_dir, 'persp_seq'))
        seq_merge = merge_seq(seq_out)
        merge_imgs.update({'persp':merge_seq(seq_out)})
        print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

        transform = build_data_transforms(['erase_1.0'])
        seq_out = transform(seq_in.copy())
        save_seq(seq_out, seq_dir=osp.join(save_dir, 'erase_seq'))
        seq_merge = merge_seq(seq_out)
        merge_imgs.update({'erase':merge_seq(seq_out)})
        print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

        transform = build_data_transforms(['flip_1.0'])
        seq_out = transform(seq_in.copy())
        save_seq(seq_out, seq_dir=osp.join(save_dir, 'flip_seq'))
        seq_merge = merge_seq(seq_out)
        merge_imgs.update({'flip':merge_seq(seq_out)})
        print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

        transform = build_data_transforms(['crop_1.0'], resolution=resolution)
        seq_out = transform(seq_in.copy())
        save_seq(seq_out, seq_dir=osp.join(save_dir, 'crop_seq'))
        seq_merge = merge_seq(seq_out)
        merge_imgs.update({'crop':merge_seq(seq_out)})
        print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

        transform = build_data_transforms(['persp_1.0', 'flip_1.0', 'rotate_1.0', 'erase_1.0', 'crop_1.0'])
        seq_out = transform(seq_in.copy())
        save_seq(seq_out, seq_dir=osp.join(save_dir, 'all_seq'))
        seq_merge = merge_seq(seq_out)
        merge_imgs.update({'all':merge_seq(seq_out)})
        print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

        rows = 1
        columns = len(merge_imgs)
        fig = plt.figure()
        merge_imgs_keys = list(merge_imgs.keys())
        for i in range(1, rows*columns+1):
            ax = fig.add_subplot(rows, columns, i)
            key = merge_imgs_keys[i-1]
            ax.set_title(key)
            plt.imshow(merge_imgs[key], cmap = plt.get_cmap('gray'))
        # plt.show()
        merge_img_name = osp.join(save_dir, 'merge_seq.png')
        plt.savefig(merge_img_name)
        plt.close("all")

