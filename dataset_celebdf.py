import os
import os.path
import random
import itertools

import torch
import torchvision
import torchvision.transforms as transforms

# from torch.utils.data import DataLoader

class celeb_df(torch.utils.data.Dataset):
    def __init__(self, directory, clip_len=64):
        super(celeb_df).__init__()

        self.clip_len = clip_len
        self.frame_transform = transforms.Compose([transforms.Resize(size=(256, 256), antialias=True)])

        self.samples = []
        self.labels = []

        # Set random seed for reproducibility
        random.seed(42)

        # Make sure only the videos are in the directory
        videos_list = os.listdir(directory)
        for video in videos_list:
            video_path = os.path.join(directory, video)

            # Get video object
            video = torchvision.io.VideoReader(video_path, "video")
            metadata = video.get_metadata()

            # Get video frames
            video_frames = []
            max_seek = metadata["video"]['duration'][0] - (self.clip_len / metadata["video"]['fps'][0])
            start = random.uniform(0., max_seek)
            start -= start % (1/metadata["video"]['fps'][0])
            for frame in itertools.islice(video.seek(start), self.clip_len):
                video_frames.append(self.frame_transform(frame['data']))
                # current_timestamp = frame['pts']
            
            # Stack frames to get (3, 64, 256, 256)
            video_out = torch.stack(video_frames, 1)

            # Append to list of samples and labels
            self.samples.append(video_out)
            if len(video_path.split("\\")[-1].split("_")) == 3:
                self.labels.append(1)
            else:
                self.labels.append(0)
        
        ### print a sample (i.e. 64 frames) and its label to test if needed ###
        # print(self.samples[5])
        # print(self.labels[5])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return (self.samples[index], self.labels[index])