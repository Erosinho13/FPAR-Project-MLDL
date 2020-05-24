from torchvision.datasets import VisionDataset
from PIL import Image
from math import ceil
import numpy as np
import random
import os
import sys
import torch

IMAGE = 0
LABEL = 1
TEST_USER = 'S2'
# directory containing the x-flows frames
FLOW_X_FOLDER = "flow_x_processed"
# directory containing the y-flows frames
FLOW_Y_FOLDER = "flow_y_processed"
# directory containing the rgb frames
FRAME_FOLDER = "processed_frames2"


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # functions that loads an image as an rgb pil object
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def flow_pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # functions that loads an image as a gray-scale pil object
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


class GTEA61(VisionDataset):
    # this class inherites from VisionDataset and represents the rgb frames of the dataset
    def __init__(self, root, split='train', seq_len=16, transform=None, target_transform=None, label_map=None):
        super(GTEA61, self).__init__(root, transform=transform, target_transform=target_transform)
        self.datadir = root
        # split indicates whether we should load the train or test split
        self.split = split
        # seq len tells us how many frames for each video we are going to consider
        # frames will be taken uniformly spaced
        self.seq_len = seq_len
        self.label_map = label_map
        if label_map is None:
            # if the label map dictionary is not provided, we are going to build it
            self.label_map = {}
        # videos is a list containing for each video, its path where you can find all its frames
        self.videos = []
        # labels[i] contains the class ID of the i-th video
        self.labels = []
        # n_frames[i] contains the number of frames available for i-th video
        self.n_frames = []

        # we expect datadir to be GTEA61, so we add FRAME_FOLDER to get to the frames
        frame_dir = os.path.join(self.datadir, FRAME_FOLDER)
        users = os.listdir(frame_dir)
        if len(users) != 4:
            raise FileNotFoundError("you specified the wrong directory")
        if TEST_USER not in users:
            raise FileNotFoundError("S2 folder not found")
        if self.split == 'test':
            folders = [users[users.index(TEST_USER)]]
        else:
            users.remove(TEST_USER)
            folders = users

        # folders is a list that contains either :
        #   - 1 element -> the path of the folder of the user S2 if split == 'test'
        #   - 3 elements -> the paths of the folders for S1,S3,S4 if split == 'train'

        if label_map is None:
            # now we build the label map; we take folders[0] just to get all class names
            # since it is GUARANTEED that all users have same classes
            classes = os.listdir(os.path.join(frame_dir, folders[0]))
            self.label_map = {act: i for i, act in enumerate(classes)}
        for user in folders:
            user_dir = os.path.join(frame_dir, user)
            # user dir it's gonna be ../GTEA61/processed_frames2/S1 or any other user
            for action in os.listdir(user_dir):
                action_dir = os.path.join(user_dir, action)
                # inside an action dir we can have 1 or more videos
                for element in os.listdir(action_dir):
                    # we add rgb to the path since there is an additional folder inside S1/1/rgb
                    # before the frames
                    frames = os.path.join(action_dir, element, "rgb")
                    # we append in videos the path
                    self.videos.append(frames)
                    # in labels the label, using the label map
                    self.labels.append(self.label_map[action])
                    # in frames its length in number of frames
                    self.n_frames.append(len(os.listdir(frames)))

    def __getitem__(self, index):
        # firstly we retrieve the video path, label and num of frames
        vid = self.videos[index]
        label = self.labels[index]
        length = self.n_frames[index]
        if self.transform is not None:
            # this is needed to randomize the parameters of the random transformations
            self.transform.randomize_parameters()

        # sort the list of frames since the name is like rgb002.png
        # so we use the last number as an ordering
        frames = np.array(sorted(os.listdir(vid)))
        # now we take seq_len equally spaced frames between 0 and length
        # linspace with the option int will give us the indices to take
        select_indices = np.linspace(0, length, self.seq_len, endpoint=False, dtype=int)
        # we then select the frames using numpy fancy indexing
        # note that the numpy arrays are arrays of strings, containing the file names
        # nevertheless, numpy will work with string arrays as well
        select_frames = frames[select_indices]
        # append to each file its path
        select_files = [os.path.join(vid, frame) for frame in select_frames]
        # use pil_loader to get pil objects
        sequence = [pil_loader(file) for file in select_files]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            sequence = [self.transform(image) for image in sequence]

        return sequence
        sequence = torch.stack(sequence, 0)

        return sequence, label

    def __len__(self):
        return len(self.videos)


class GTEA61_flow(VisionDataset):
    def __init__(self, root, split='train', seq_len=5, transform=None, target_transform=None, label_map=None):
        super(GTEA61_flow, self).__init__(root, transform=transform, target_transform=target_transform)
        self.datadir = root
        self.split = split
        self.seq_len = seq_len
        self.label_map = label_map
        if label_map is None:
            self.label_map = {}
        self.x_frames = []
        self.y_frames = []
        self.labels = []
        self.n_frames = []

        flow_dir = os.path.join(self.datadir, FLOW_X_FOLDER)
        users = os.listdir(flow_dir)
        if len(users) != 4:
            raise FileNotFoundError("you specified the wrong directory")
        if TEST_USER not in users:
            raise FileNotFoundError("S2 folder not found")
        if self.split == 'test':
            folders = [users[users.index(TEST_USER)]]
        else:
            users.remove(TEST_USER)
            folders = users

        classes = os.listdir(os.path.join(flow_dir, folders[0]))
        if label_map is None:
            self.label_map = {act: i for i, act in enumerate(classes)}
        for user in folders:
            user_dir = os.path.join(flow_dir, user)
            for action in os.listdir(user_dir):
                action_dir = os.path.join(user_dir, action)
                for element in os.listdir(action_dir):
                    frames = os.path.join(action_dir, element)
                    self.x_frames.append(frames)
                    self.y_frames.append(frames.replace(FLOW_X_FOLDER, FLOW_Y_FOLDER))
                    self.labels.append(self.label_map[action])
                    self.n_frames.append(len(os.listdir(frames)))

    def __getitem__(self, index):
        vid_x = self.x_frames[index]
        vid_y = self.y_frames[index]
        label = self.labels[index]
        length = self.n_frames[index]
        self.transform.randomize_parameters()

        frames_x = np.array(sorted(os.listdir(vid_x)))
        frames_y = np.array(sorted(os.listdir(vid_y)))
        if self.split == 'train':
            startFrame = random.randint(0, length - self.seq_len)
        else:
            startFrame = np.ceil((length - self.seq_len) / 2)
        select_indices = startFrame + np.arange(0, self.seq_len)
        select_x_frames = frames_x[select_indices]
        select_y_frames = frames_y[select_indices]
        select_frames = np.ravel(np.column_stack((select_x_frames, select_y_frames)))

        select_files = [os.path.join(vid_x, frame) for frame in select_frames]
        select_files[1::2] = [y_files.replace('x','y') for y_files in select_files[1::2]]
        sequence = [flow_pil_loader(file) for file in select_files]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            sequence[::2] = [self.transform(image, inv=True, flow=True) for image in sequence[::2]]
            sequence[1::2] = [self.transform(image, inv=False, flow=True) for image in sequence[1::2]]
        sequence = torch.stack(sequence, 0).squeeze(1)

        return sequence, label

    def __len__(self):
        return len(self.x_frames)


class GTEA61_2Stream(VisionDataset):
    def __init__(self, root, split='train', seq_len=7, stack_size=5, transform=None, target_transform=None):
        super(GTEA61_2Stream, self).__init__(root, transform=transform, target_transform=target_transform)
        self.datadir = root
        self.split = split
        self.seq_len = seq_len
        self.stack_size = stack_size

        frame_dir = os.path.join(self.datadir, FRAME_FOLDER)
        users = os.listdir(frame_dir)
        if len(users) != 4:
            raise FileNotFoundError("you specified the wrong directory")
        if TEST_USER not in users:
            raise FileNotFoundError("S2 folder not found")
        if self.split == 'test':
            folders = [users[users.index(TEST_USER)]]
        else:
            users.remove(TEST_USER)
            folders = users
        classes = os.listdir(os.path.join(frame_dir, folders[0]))
        self.label_map = {act: i for i, act in enumerate(classes)}
        self.frame_dataset = GTEA61(self.datadir, split=self.split, seq_len=self.seq_len,
                                    transform=self.transform, label_map=self.label_map)
        self.flow_dataset = GTEA61_flow(self.datadir, split=self.split, seq_len=self.stack_size,
                                        transform=self.transform, label_map=self.label_map)

    def __getitem__(self, index):
        frame_seq, label = self.frame_dataset.__getitem__(index)
        flow_seq, _ = self.flow_dataset.__getitem__(index)
        return flow_seq, frame_seq, label

    def __len__(self):
        return self.frame_dataset.__len__()
