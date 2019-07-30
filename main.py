from train import *
import argparse

# reshaped_dir = os.getcwd() + "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_reshaped"
# landmark_dir = os.getcwd() + "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_landmarks"
# overlay_dir = os.getcwd() + "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_landmarks_overlayed"

parser = argparse.ArgumentParser(description="Main file for talking faces")
parser.add_argument('--load_path', type=str, default="no_path", help = "path to load in checkpoint")
parser.add_argument('--test', action = 'store_true', help = "specify to test instead of train")
parser.add_argument('--num_epochs', type = int, default=1, help = "number of epochs")
parser = parser.parse_args()

loader = None
if parser.load_path != "no_path":
	loader = Loader(load_path)
	checkpoints = loader.loadModel()



if parser.test == False:
	reshaped_dir = "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_reshaped"
	landmark_dir = "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_landmarks"
	overlay_dir = "/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/data/youtube_aligned/aligned_images_DB_landmarks_overlayed"

	reshaped_d, reshaped_sequence_count, reshaped_sequences = get_data_dict(reshaped_dir)
	landmark_d, landmark_sequence_count, landmark_sequences = get_data_dict(landmark_dir)

	reshaped_sequence_count, reshaped_dict, landmark_dict, num_sequences = readImages(reshaped_sequence_count,reshaped_sequences,landmark_sequences)

	print("Successfully loaded images and landmark images")
	composer = Composer(projection_dims=256,embedding_dims=256,reshaped_sequences=reshaped_dict,landmark_sequences=landmark_dict,
	                        landmark_sequence_count=reshaped_sequence_count,num_sequences=num_sequences)
	composer.metaTrain(parser.num_epochs)
