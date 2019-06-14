import numpy as np
import skimage
import cv2
from data_processing import *

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MetaDataset(Dataset):
    def __init__(self,reshaped_frame_sequences,landmark_frame_sequences,num_videos,k):
        """
            Args
        """
        self.reshaped_frame_sequences = reshaped_frame_sequences
        self.landmark_frame_sequences = landmark_frame_sequences
        self.video_sequence_names = sorted(list(self.reshaped_frame_sequences.keys()))
        self.landmark_sequence_names = sorted(list(self.landmark_frame_sequences.keys()))
        self.num_videos = num_videos
        self.k = k
        
    def __len__(self):
        return len(self.video_sequence_names)

    def __getitem__(self,index):
        reshaped_frame_sequence = self.reshaped_frame_sequences[self.video_sequence_names[index]]
        landmark_frame_sequence = self.landmark_frame_sequences[self.landmark_sequence_names[index]]
    
        sequence_indices = range(len(reshaped_frame_sequence))
        target_index = random.choice(sequence_indices) 
        sequence_indices = [i for i in sequence_indices if i != target_index]
        
        sampled_vids = []
        if self.k >= len(sequence_indices):
            for i in sequence_indices:
                sampled_vids.append((reshaped_frame_sequence[i],landmark_frame_sequence[i]))
        else:
            sampled_sequence_indices = random.sample(sequence_indices,self.k)
            for i in sampled_sequence_indices:
                sampled_vids.append((reshaped_frame_sequence[i],landmark_frame_sequence[i]))
        target = (reshaped_frame_sequence[target_index],landmark_frame_sequence[target_index])
        return target , sampled_vids

def makeDataloader(dataset,batch_size=1,shuffle=False,drop_last=True):
	cuda = torch.cuda.is_available()
	kwargs = {'num_workers':3, 'pin_memory':True} if cuda else {}
	print ("gpu available :", cuda)
	device = torch.device("cuda" if cuda else "cpu")
	num_gpu = torch.cuda.device_count()
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
	return dataloader