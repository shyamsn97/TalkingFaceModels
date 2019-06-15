from layers import *
from models import *
from data_processing import *
from loss import *
from vgg import *
from Dataset import *
from utils import *
from train import *


# reshaped_dir = os.getcwd() + "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_reshaped"
# landmark_dir = os.getcwd() + "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_landmarks"
# overlay_dir = os.getcwd() + "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_landmarks_overlayed"

reshaped_dir = "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_reshaped"
landmark_dir = "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_landmarks"
overlay_dir = "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_landmarks_overlayed"

reshaped_d, reshaped_sequence_count, reshaped_sequences = get_data_dict(reshaped_dir)
landmark_d, landmark_sequence_count, landmark_sequences = get_data_dict(landmark_dir)

reshaped_sequence_count, reshaped_dict, landmark_dict, num_sequences = readImages(reshaped_sequence_count,reshaped_sequences,landmark_sequences)

print("Successfully loaded images and landmark images")
meta = MetaLearningStage(projection_dims=128,embedding_dims=128,reshaped_sequences=reshaped_dict,landmark_sequences=landmark_dict,
                        landmark_sequence_count=reshaped_sequence_count,num_sequences=num_sequences)
meta.train(5)
