from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from os.path import exists, join
from os import listdir

import random
import numpy as np
from torch.utils.data import Dataset

import torch
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
except:
    from PIL import Image as InterpolationMode
from .transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupRandomHorizontalFlip


def to_RGB(image):
    return image.convert("RGB")


class RetrievalDataset(Dataset):
    """General dataset."""

    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=32,
            max_frames=12,
            video_framerate=1,
            image_resolution=224,
            slice_framepos=2, # "0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly."
            mode='all',
            config=None
    ):
        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        self.anno_path = anno_path
        self.video_path = video_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.video_framerate = video_framerate
        self.image_resolution = image_resolution
        self.slice_framepos = slice_framepos
        print('############################### self.slice_framepos ############################', self.slice_framepos)
        self.mode = mode  # all/text/vision
        self.config = config
        self.train_augment = config.train_augment
        self.hflip = config.horizontal_flip

        self.image_dict, self.sentences_dict = self._get_anns(self.subset)

        self.video_list = list(self.image_dict.keys())
        self.sample_len = 0

        print("Video number: {}".format(len(self.image_dict)))
        print("Total Pairs: {}".format(len(self.sentences_dict)))

        if self.subset == 'train' and self.train_augment:

            if self.hflip:
                print("Using strong training augmentation with horizontal flip!")

                self.transform = Compose([
                    GroupMultiScaleCrop(image_resolution, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
            else:
                print("Using strong training augmentation without horizontal flip!")

                self.transform = Compose([
                    GroupMultiScaleCrop(image_resolution, [1, .875, .75, .66]),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])

        else:
            self.transform = Compose([
                Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_resolution),
                to_RGB,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.image_resolution = image_resolution

        if self.subset == 'train' and (not self.config.expand_msrvtt_sentences):
            self.sample_len = len(self.csv)
        elif self.mode in ['all', 'text']:
            self.sample_len = len(self.sentences_dict)
        else:
            self.sample_len = len(self.video_list)
        print(self.subset, self.mode, self.sample_len)

    def __len__(self):
        return self.sample_len if not self.config.debug else 20

    def _get_anns(self, subset='train'):
        raise NotImplementedError

    def _get_text(self, caption):
        if len(caption) == 3:
            _caption_text, s, e = caption
        else:
            raise NotImplementedError

        if isinstance(_caption_text, list):
            if self.config.no_expand_type == 'first':
                _caption_text = _caption_text[0]
            elif self.config.no_expand_type == 'rand':
                _caption_text = random.choice(_caption_text)

        caption_text_list = _caption_text if isinstance(_caption_text, list) else [_caption_text]

        input_ids, input_mask = [], []
        for caption_text in caption_text_list:
            _input_ids, _input_mask = self.proc_text(caption_text)
            input_ids.append(_input_ids)
            input_mask.append(_input_mask)

        input_ids = np.array(input_ids).squeeze()
        input_mask = np.array(input_mask).squeeze()
        return input_ids, input_mask, s, e

    def proc_text(self, caption_text):
        words = self.tokenizer.tokenize(caption_text)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        return input_ids, input_mask

    def _get_rawvideo_dec(self, video_id, s=None, e=None):
        # speed up video decode via decord.
        video_mask = np.zeros(self.max_frames, dtype=np.long)
        max_video_length = 0

        # T x 3 x H x W
        video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
        video_path = self.video_dict[video_id]

        if exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            print(video_path)
            raise FileNotFoundError

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            # T x 3 x H x W
            sample_fps = int(self.video_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > self.max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
            else:
                sample_pos = all_pos

            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
            patch_images = torch.stack([self.transform(img) for img in patch_images])
            slice_len = patch_images.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = patch_images
        else:
            print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def _get_capimage(self, video_id):

        # speed up video decode via decord.
        video_mask = np.zeros(self.max_frames, dtype=np.long)
        max_video_length = 0
        # T x 3 x H x W
        video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float)
        image_folder = self.image_dict[video_id]
        # if not exists(image_folder):
        #     print(image_folder)
        #     print(video_id)
        #     raise FileNotFoundError
        image_path_list = listdir(image_folder)
        image_path_list.sort()

        slice_len = len(image_path_list)
        # print('slice_len', slice_len)
        # print('slice_framepos', self.slice_framepos)
        # print('self.max_frames', self.max_frames)
        # print('image_path_list', image_path_list)
        max_video_length = max_video_length if max_video_length > slice_len else slice_len
        if slice_len < 1:
            pass
        elif slice_len > self.max_frames:
            # video = patch_images[:self.max_frames, ...]
            if self.slice_framepos == 0:    # 0: cut from head frames
                sample_img_list = image_path_list[:self.max_frames]
            elif self.slice_framepos == 1:  # 1: cut from tail frames
                sample_img_list = image_path_list[-self.max_frames:]
            else:                           # 2: extract frames uniformly
                sample_indx = list(np.linspace(0, slice_len - 1, num=self.max_frames, dtype=int))
                # print('sample_indx', sample_indx)
                sample_img_list = [image_path_list[tmp_indx] for tmp_indx in sample_indx]
        else:
            sample_img_list = image_path_list

        # print('sample_img_list', sample_img_list)

        sample_list = []
        # for p in image_path_list[:self.max_frames]:
        for p in sample_img_list:
            sample_list.append(np.array(Image.open(join(image_folder, p))))
        sample_batch = np.stack(sample_list, axis=0)
        if self.subset == 'train' and self.train_augment:
            patch_images = [Image.fromarray(f).convert('RGB') for f in sample_batch]
            patch_images = self.transform(patch_images)
            _, H, W = patch_images.size()
            patch_images = patch_images.reshape(-1, 3, H, W)
        else:
            patch_images = [Image.fromarray(f) for f in sample_batch]
            patch_images = torch.stack([self.transform(img) for img in patch_images]) # T, C, H, W

        if slice_len > self.max_frames:
            video = patch_images
        else:
            video[:slice_len, ...] = patch_images

        # sample_list = []
        # # for p in image_path_list[:self.max_frames]:
        # for p in image_path_list:
        #     sample_list.append(np.array(Image.open(join(image_folder, p))))
        # sample_batch = np.stack(sample_list, axis=0)
        # if self.subset == 'train' and self.train_augment:
        #     patch_images = [Image.fromarray(f).convert('RGB') for f in sample_batch]
        #     patch_images = self.transform(patch_images)
        #     _, H, W = patch_images.size()
        #     patch_images = patch_images.reshape(-1, 3, H, W)
        # else:
        #     patch_images = [Image.fromarray(f) for f in sample_batch]
        #     patch_images = torch.stack([self.transform(img) for img in patch_images]) # T, C, H, W
        # slice_len = patch_images.shape[0]
        # max_video_length = max_video_length if max_video_length > slice_len else slice_len
        # print('slice_len', slice_len)
        # print('self.max_frames', self.max_frames)
        # print('self.slice_framepos', self.slice_framepos)
        # if slice_len < 1:
        #     pass
        # elif slice_len > self.max_frames:
        #     print(self.slice_framepos)
        #     # video = patch_images[:self.max_frames, ...]
        #     if self.slice_framepos == 0:    # 0: cut from head frames
        #         video = patch_images[:self.max_frames, ...]
        #     elif self.slice_framepos == 1:  # 1: cut from tail frames
        #         video = patch_images[-self.max_frames:, ...]
        #     else:                           # 2: extract frames uniformly
        #         sample_indx = np.linspace(0, patch_images.shape[0] - 1, num=self.max_frames, dtype=int)
        #         video = patch_images[sample_indx, ...]
        # else:
        #     video[:slice_len, ...] = patch_images

        if slice_len > self.max_frames:
            video_mask[:self.max_frames] = [1] * self.max_frames
        else:
            video_mask[:max_video_length] = [1] * max_video_length
        if isinstance(video, torch.Tensor):
            video = video.numpy()

        return video, video_mask

    def __getitem__(self, idx):

        if self.mode == 'all':
            if (self.subset == 'train' and not self.config.expand_msrvtt_sentences) or \
                    (self.subset != 'train' and self.config.mq_test):  # 多标签
                video_id, caption = self.csv['video_id'].values[idx], None
                text_ids, text_mask, s, e = self._get_text((self.sentences[video_id], None, None))
            else:
                video_id, caption = self.sentences_dict[idx]
                text_ids, text_mask, s, e = self._get_text(caption)
            # video, video_mask = self._get_rawvideo_dec(video_id, s, e)
            video, video_mask = self._get_capimage(video_id)
            # video, video_mask = self._get_rawvideo(video_id, s, e)
            return text_ids, text_mask, np.zeros_like(text_ids), video, video_mask, video_id
        elif self.mode == 'text':
            video_id, caption = self.sentences_dict[idx]
            text_ids, text_mask, s, e = self._get_text(caption)
            return text_ids, text_mask, idx
        elif self.mode == 'video':
            video_id = self.video_list[idx]
            # video, video_mask = self._get_rawvideo_dec(video_id)
            video, video_mask = self._get_capimage(video_id)
            # video, video_mask = self._get_rawvideo(video_id)
            return video, video_mask, idx

    def get_text_len(self):
        return len(self.sentences_dict)

    def get_video_len(self):
        return len(self.video_list)

    def get_text_content(self, ind):
        return self.sentences_dict[ind][1]

    def get_data_name(self):
        return self.__class__.__name__ + "_" + self.subset
