import json
import os
import os.path as osp
from pathlib import Path as P
from itertools import chain as chain
import time
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from fvcore.common.file_io import PathManager
from einops import rearrange, reduce, repeat
from tqdm import tqdm

from .. import autoaugment as autoaugment
from ...utils import logging
from ...utils import distributed as du
from .. import utils as dataset_utils
from ..build import DATASET_REGISTRY

import ffmpeg
from collections import defaultdict
import cv2

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Ssv2_webm(Dataset):
    """
    ssv2 dataset for TimeSformer models. Uses webm files.
    """
    def __init__(self, cfg, mode, num_retries=10):
        """
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Something-Something V2".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.ERM_TEST.NUM_ENSEMBLE_VIEWS * cfg.ERM_TEST.NUM_SPATIAL_CROPS
            )

        if du.is_master_proc(du.get_world_size()):
            logger.info("Constructing Something-Something V2 {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == 'train':
            data_split = 'train'
            split_dir = self.cfg.DATA.TRAIN_SPLIT_DIR
        
        else:
            # Note that we do not consider 'testset' for ssv2, as it does not come with labels.
            data_split = 'val'
            split_dir = self.cfg.DATA.VAL_SPLIT_DIR

        videos_dir = osp.join(self.cfg.DATA.PATH_TO_JPEG, split_dir)

        # Loading label names.
        with PathManager.open(
            osp.join(
                self.cfg.DATA.PATH_TO_ANNOTATION,
                "something-something-v2-labels.json",
            ),
            "r",
        ) as f:
            label_dict = json.load(f)

        # Loading labels.
        label_file = osp.join(
            self.cfg.DATA.PATH_TO_ANNOTATION,
            "something-something-v2-{}.json".format(
                "train" if self.mode == "train" else "validation"
            ),
        )
        with PathManager.open(label_file, "r") as f:
            label_json = json.load(f)

        self._video_names = []
        self._labels = []
        for video in label_json:
            video_name = video["id"]
            template = video["template"]
            template = template.replace("[", "")
            template = template.replace("]", "")
            label = int(label_dict[template])
            self._video_names.append(video_name)
            self._labels.append(label)

        self._path_to_videos = [osp.join(videos_dir, f"{name}.webm") for name in self._video_names]

        # Extend self when self._num_clips > 1 (during testing).
        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._video_names = list(
            chain.from_iterable([[x] * self._num_clips for x in self._video_names])
        )

        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(len(self._path_to_videos))
                ]
            )
        )

        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )

        logger.info(
            "Something-Something V2 dataset constructed "
            " (size: {})".format(
                len(self._path_to_videos)
            )
        )
    
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        label = self._labels[index]
        video_name = self._video_names[index]
        video_path = self._path_to_videos[index]

        if self.mode in ["train", "val"]: #or self.cfg.MODEL.ARCH in ['resformer', 'vit']:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.ERM_TEST.NUM_SPATIAL_CROPS
            )
            if self.cfg.ERM_TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1

            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1

        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        num_frames = self.cfg.DATA.NUM_FRAMES
        probe = ffmpeg.probe(video_path)
        
        stream_dict = probe['streams'][0]
        format_dict = probe['format']
        width, height = stream_dict['width'], stream_dict['height'] #, stream_dict['pix_fmt']

        stream = ffmpeg.input(video_path, vsync='0')
        out, info = stream.output('pipe:', format='rawvideo', pix_fmt='rgb24', v='trace').run(capture_stdout=True, capture_stderr=True)
        
        frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3]) # frames in np.array
        video_length = len(frames)
        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        frames = torch.as_tensor(frames[seq])
        if self.cfg.DATA.USE_RAND_AUGMENT and self.mode in ["train"]:
            # Transform to PIL Image
            frames = [transforms.ToPILImage()(frame.squeeze().numpy()) for frame in frames]

            # Perform RandAugment
            img_size_min = crop_size
            auto_augment_desc = "rand-m20-mstd0.5-inc1"
            aa_params = dict(
                translate_const=int(img_size_min * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in self.cfg.DATA.MEAN]),
            )
            seed = random.randint(0, 100000000)
            frames = [autoaugment.rand_augment_transform(
                auto_augment_desc, aa_params, seed)(frame) for frame in frames]

            # To Tensor: T H W C
            frames = [torch.tensor(np.array(frame)) for frame in frames]
            frames = torch.stack(frames)

        frames = dataset_utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        frames = rearrange(frames, 't h w c -> t c h w')
        frames, random_samples = dataset_utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            return_random_samples=True
        )

        length, channels, height, width = frames.size(0), frames.size(1), frames.size(2), frames.size(3)
        shape = torch.as_tensor([length, channels, height, width])
        video_name = torch.as_tensor(int(video_name))
        video_index = torch.as_tensor(index)
        time_index = torch.as_tensor(seq)
        spatial_sample_index = torch.as_tensor(spatial_sample_index)

        output_feature = frames.squeeze() # remove dummy time dimension (if exists).

        if self.mode in ["train"]:
            gray_frames = (0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]).numpy()
            flows = []
            for t in range(len(gray_frames)-1):
                flow = cv2.calcOpticalFlowFarneback(gray_frames[t]*128,gray_frames[t+1]*128, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_patch = rearrange(flow,'(h p1) (w p2) c -> c h w (p1 p2)', p1 = self.cfg.VIT.PATCH_SIZE, p2 = self.cfg.VIT.PATCH_SIZE)
                flow_patch = flow_patch.mean(-1)
                mag, ang = cv2.cartToPolar(flow_patch[0], flow_patch[1])
                ang = ang*180/np.pi
                mag = cv2.normalize(mag, None, 0, 100, cv2.NORM_MINMAX).reshape(-1)//1 > 0
                flows.append(torch.as_tensor((np.clip(ang//45, 0, 7)+1).reshape(-1) * mag).long())
            flows = torch.stack(flows)
            return output_feature, label, {"flow": flows, "spatial_sample_index": spatial_sample_index, "random_samples": random_samples, "shape": shape, "time_index": time_index, "video_name": video_name, "video_index": video_index}
    
        else:
            return output_feature, label, {"spatial_sample_index": spatial_sample_index, "random_samples": random_samples, "shape": shape, "time_index": time_index, "video_name": video_name, "video_index": video_index}
    
    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
