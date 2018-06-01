import os
import types
import json
import numpy as np
import torch
import torch.utils.data.dataset

from PIL import Image

# Local import
import video

# Take random time slice from video tensor
class RandomTimeSliceVideo:
    def __init__(self, transform, num_frames=16):
        self.transform = transform
        self.num_frames = num_frames

    def __call__(self, vid_path):
        tot_frames, width, height, framerate = video.get_info(vid_path)
        t0 = np.random.randint(0, tot_frames-self.num_frames)
        vid_arr = video.load_clip(vid_path, tot_frames, framerate, t0, self.num_frames, width, height)
        tsr_list = []
        for arr in vid_arr:
            tsr = torch.FloatTensor(arr)
            tsr = self.transform(tsr).unsqueeze(0)
            tsr_list.append(tsr)
        full_tsr = torch.cat(tsr_list)
        return full_tsr

# Take random time slice from video tensor
class TimeSliceVideoSet:
    def __init__(self, transform, num_frames=16):
        self.transform = transform
        self.num_frames = num_frames

    def __call__(self, vid_path):
        tot_frames, width, height, framerate = video.get_info(vid_path)
        slice_list = []
        for i in range(0, tot_frames-self.num_frames, self.num_frames):
            vid_arr = video.load_clip(vid_path, tot_frames, framerate, i, self.num_frames, width, height)
            tsr_list = []
            for arr in vid_arr:
                tsr = torch.FloatTensor(arr)
                tsr = self.transform(tsr).unsqueeze(0)
                tsr_list.append(tsr)
            slice_tsr = torch.cat(tsr_list).unsqueeze(0)
            slice_list.append(slice_tsr)
        slice_set = torch.cat(slice_list)
        return slice_set

# Take random time slice from video tensor
class RandomTimeSliceTensor:
    def __init__(self, transform, num_frames=16):
        self.transform = transform
        self.num_frames = num_frames

    def __call__(self, tsr_dir):
        file_list = sorted(os.listdir(tsr_dir))
        t0 = np.random.randint(0, len(file_list)-self.num_frames)
        tsr_list = []
        for tsr_file in file_list[t0:t0+self.num_frames]:
            tsr_path = os.path.join(tsr_dir, tsr_file)
            tsr = torch.load(tsr_path)
            tsr = self.transform(tsr).unsqueeze(0)
            tsr_list.append(tsr)
        full_tsr = torch.cat(tsr_list)
        return full_tsr

# Take random time slice from video tensor
class RandomTimeSliceImage:
    def __init__(self, transform, num_frames=16):
        self.transform = transform
        self.num_frames = num_frames

    def __call__(self, tsr_dir):
        file_list = sorted(os.listdir(tsr_dir))
        t0 = np.random.randint(0, len(file_list)-self.num_frames)
        img_list = []
        for tsr_file in file_list[t0:t0+self.num_frames]:
            tsr_path = os.path.join(tsr_dir, tsr_file)
            tsr = torch.load(tsr_path)
            img = self.transform(tsr)
            img_list.append(img)
        return img_list

# Take random time slice from video tensor
class TimeSliceSet:
    def __init__(self, transform, num_frames=16):
        self.transform = transform
        self.num_frames = num_frames

    def __call__(self, tsr_dir):
        file_list = sorted(os.listdir(tsr_dir))
        slice_list = []
        for i in range(0, len(file_list)-self.num_frames, self.num_frames):
            tsr_list = []
            for j in range(i, i+self.num_frames, 1):
                tsr_file = file_list[j]
                tsr_path = os.path.join(tsr_dir, tsr_file)
                tsr = torch.load(tsr_path)
                new_tsr = self.transform(tsr).unsqueeze(0)
                tsr_list.append(new_tsr)
            slice_tsr = torch.cat(tsr_list).unsqueeze(0)
            slice_list.append(slice_tsr)
        slice_set = torch.cat(slice_list)
        return slice_set

class VideoDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, tensor_dir, video_dir, raw_path, mode, 
            transform=None, slice_transform=None, num_frames=None, eval_mode=False, num_slices=None, preload=False, storage='tensor'):
        self.preload = preload
        self.num_slices = num_slices
        self.transform = transform
        self.slice_transform = slice_transform
        self.eval_mode = eval_mode
        if eval_mode:
            if storage == 'tensor':
                self.time_slice = TimeSliceSet(self.transform, num_frames=num_frames)
            elif storage == 'video':
                self.time_slice = TimeSliceVideoSet(self.transform, num_frames=num_frames)
        else:
            if storage == 'tensor':
                self.time_slice = RandomTimeSliceTensor(self.transform, num_frames=num_frames)
            elif storage == 'video':
                self.time_slice = RandomTimeSliceVideo(self.transform, num_frames=num_frames)
        ###
        """
        rt_class_list, rt_score_list = [], []
        blacklist_idx = []
        with open(rt_path, 'r') as fp:
            rt_data = json.load(fp)
            for i, d in enumerate(rt_data):
                try:
                    c, s = d
                except:
                    blacklist_idx.append(i)
                else:
                    rt_class_list.append(c)
                    rt_score_list.append(s)
        ###
        tag_list = []
        with open(tag_path, 'r') as fp:
            for i, line in enumerate(fp):
                if i not in blacklist_idx:
                    tag_list.append(line.strip())
        ###
        tsr_list = []
        rt_list = []
        for tag, rt in zip(tag_list, rt_class_list):
            tsr_path = os.path.join(tensor_dir, tag)
            if os.path.isdir(tsr_path):
                tsr_list.append(tsr_path)
                rt_list.append(rt)
        ###
        """
        with open(raw_path, 'r') as fp:
            raw_dict = json.load(fp)
        dup_list = sum([v for v in raw_dict.values()], [])
        tag_set = set()
        tsr_list, rt_list = [], []
        print(len(dup_list))
        for d in sorted(dup_list, key=lambda x: x['title']):
            tag = os.path.basename(d['mainTrailer']['sourceId'])
            if tag not in tag_set and len(tag) > 0:
                tag_set.add(tag)
                # Get tensor dir or video file path
                if storage == 'tensor':
                    tsr_path = os.path.join(tensor_dir, tag)
                elif storage == 'video':
                    tsr_path = os.path.join(video_dir, '{}.mp4'.format(tag))
                # Check if path exists
                if os.path.exists(tsr_path):
                    tom_class = d['tomatoIcon']
                    pop_class = d['popcornIcon']
                    
                    if tom_class == 'fresh' or tom_class == 'certified_fresh':
                        tom_val = 1
                    elif tom_class == 'rotten':
                        tom_val = 0
                    elif tom_class == 'NA':
                        continue
                    else:
                        raise Exception('Invalid tom_class: {}'.format(tom_class))

                    if pop_class == 'upright':
                        pop_val = 1
                    elif pop_class == 'spilled':
                        pop_val = 0
                    elif pop_class == 'NA':
                        continue
                    else:
                        raise Exception('Invalid pop_class: {}'.format(pop_class))

                    tsr_list.append(tsr_path)
                    rt_list.append(tom_val)
        # Seed RNG
        np.random.seed(123)
        # Shuffle data
        rand_idx = np.arange(len(tsr_list))
        np.random.shuffle(rand_idx)
        # Shuffle it
        self.tsr_arr = np.array(tsr_list, dtype=object)[rand_idx]
        self.rt_class_arr = np.array(rt_list, dtype=object)[rand_idx]
        #self.rt_score_arr = np.array(rt_score_list, dtype=object)[rand_idx]
        if mode == 'train':
            self.tsr_arr = self.tsr_arr[:-300]
            self.rt_class_arr = self.rt_class_arr[:-300]
            #self.rt_score_arr = self.rt_score_arr[:-100]
        elif mode == 'val':
            self.tsr_arr = self.tsr_arr[-300:]
            self.rt_class_arr = self.rt_class_arr[-300:]
            #self.rt_score_arr = self.rt_score_arr[-100:]
        if self.preload:
            print('==> Preloading...')
            tsr_list = []
            for i, path in enumerate(self.tsr_arr):
                print('... preloading ({})'.format(i))
                vid_tsr = torch.load(path)
                vid_tsr = torch.cat([self.transform(tsr).unsqueeze(0) for tsr in vid_tsr])
                tsr_list.append(vid_tsr)
            self.tsr_arr = tsr_list
        ###
        print(mode, len(self.tsr_arr), len(self.rt_class_arr))
        print('{} BIAS: {:.2f}'.format(mode, self.rt_class_arr.mean()))

    def __getitem__(self, index):
        if self.preload:
            vid_tsr = self.tsr_arr[index]
        else:
            #vid_tsr = torch.load(self.tsr_arr[index])
            #vid_tsr = torch.cat([self.transform(tsr).unsqueeze(0) for tsr in vid_tsr])
            tsr_dir = self.tsr_arr[index] 
        full_slice_list, full_label_list = [], []
        for i in range(self.num_slices):
            label_tsr = torch.LongTensor([self.rt_class_arr[index]]).unsqueeze(0)
            slice_tsr = self.time_slice(tsr_dir)
            if self.eval_mode:
                proc_tsr = torch.cat([self.slice_transform(s) for s in slice_tsr])
            else:
                proc_tsr = self.slice_transform(slice_tsr).unsqueeze(0)
            full_slice_list.append(proc_tsr)
            full_label_list.append(label_tsr)

        return torch.cat(full_slice_list), torch.cat(full_label_list)

    def __len__(self):
        return len(self.tsr_arr)

class FeatureDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, feature_dir, modality, mode, feat_mean, feat_std, stride):
        feature_file = '{}_{}_{}.t7'.format(modality, mode, stride)
        feature_path = os.path.join(feature_dir, feature_file)
        data_dict = torch.load(feature_path)
        self.feat_list = data_dict['features']
        self.label_list = data_dict['labels']
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def norm(self):
        feat_tsr = torch.cat(self.feat_list)
        feat_mean = torch.mean(feat_tsr, dim=0)
        feat_std = torch.std(feat_tsr, dim=0)
        feat_std[feat_std==0] = 1
        return feat_mean, feat_std

    def __getitem__(self, index):
        feat_tsr = self.feat_list[index]
        label_tsr = self.label_list[index]
        norm_tsr = (feat_tsr - self.feat_mean)/self.feat_std

        return norm_tsr, label_tsr#.repeat(len(norm_tsr))

    def __len__(self):
        return len(self.feat_list)

