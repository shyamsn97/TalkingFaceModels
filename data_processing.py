import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os 
import shutil
import dlib
import torch
import skimage
import scipy.misc
from tqdm import tqdm
import face_alignment
import random

def plot_img(img,size):
	#img = cv2.imread(path,1)
	img = cv2.resize(img,(size,size))
	plt.imshow(img)

def get_data_dict(data_path,max_val=100000):
	#data_path = directory + "/data/youtube_aligned/aligned_images_DB"
	people = glob.glob(data_path + "/**")
	d = {}
	count = 0
	sequence_d = {}
	for p in people[:max_val]:
		name = p.split("/")[-1]
		d[name] = {}
		d[name]["path"] = p
		subs = glob.glob(p + "/**")
		d[name]["sequences"] = {}
		for s in subs:
			count += 1
			imgs = glob.glob(s + "/*.jpg")
			d[name]["sequences"][s.split("/")[-1]] = imgs
			sequence_d[s] = imgs 
	return d, count, sequence_d

"""
	Landmark detection code adapted from:
	https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
"""            

def landmark_mask(img,landmarks):
	layered_img = img.copy()
	mask = np.zeros_like(img) + 255.0
	facial_map = {
		(0,17): [(255,0,255),"jaw"],
		(27,31): [(0,0,255),"nose stem"],
		(31,36): [(0,0,255),"nose"],
		(42,48): [(0,255,0),"left eye"],
		(36,42): [(0,255,0),"right eye"],
		(22,27): [(255,255,0),"left eyebrow"],
		(17,22): [(255,255,0),"right eyebrow"],
		(60,68): [(255,0,0),"inner_mouth"],
		(48,68): [(255,0,0),"mouth"]
	}
	
	for key in facial_map:
		l,h = key
		color,name = facial_map[key]
		pts = landmarks[l:h]
#         if name == 'nose stem':
#             pts += [landmarks[33]]
#         if name == 'nose':
#             pts = landmarks[31:33] + landmarks[34:36]
		if name == "jaw":
			for i in range(1, len(pts)):
				ptA = tuple(pts[i - 1])
				ptB = tuple(pts[i])
				cv2.line(mask, ptA, ptB, color, 1)
				cv2.line(layered_img, ptA, ptB, color, 1)
		else:
			hull = cv2.convexHull(np.array(pts))
			cv2.drawContours(mask, [hull], -1, color,1)
			cv2.drawContours(layered_img, [hull], -1, color,1)
	return mask , layered_img

def getLandmarks_Mask(img):
	facial_map = {
		(0,17): [(255,0,255),"jaw"],
		(27,31): [(0,0,255),"nose stem"],
		(31,36): [(0,0,255),"nose"],
		(42,48): [(0,255,0),"left eye"],
		(36,42): [(0,255,0),"right eye"],
		(22,27): [(255,255,0),"left eyebrow"],
		(17,22): [(255,255,0),"right eyebrow"],
		(60,68): [(255,0,0),"inner_mouth"],
		(48,60): [(255,0,0),"mouth"]
	}
	
	gray = img #no grayscale image
	
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("/mnt/d/ssudhakaran/Code/SamsungFace/SamsungFace/shape_predictor_68_face_landmarks.dat")
	landmark_img = np.zeros_like(img) + 255.0

	faces = detector(gray)
	if len(faces) == 0:
		return
	for face in faces:
		x1 = face.left()
		y1 = face.top()
		x2 = face.right()
		y2 = face.bottom()
		#cv2.rectangle(white_img, (x1, y1), (x2, y2), (0, 0, 0), thickness = 1)

		landmarks = predictor(gray, face)
		landmark_points = []
		point_dict = {}
		for i in range(0, 68):
			x = landmarks.part(i).x
			y = landmarks.part(i).y
			landmark_points.append(((x,y)))
			#cv2.circle(landmark_img, (x, y), 2, (0, 0, 0),thickness=-1)
		for r in facial_map.keys():
			color, typ = facial_map[r]
			point_dict[typ] = []
			for i in range(r[0],r[1]):
				x = landmarks.part(i).x
				y = landmarks.part(i).y
				point_dict[typ].append((x,y))
				cv2.circle(landmark_img, (x, y), 1, color,thickness=-1)
				
	mask, layered = landmark_mask(img,landmark_points)
	return landmark_img, mask, layered, point_dict


def processLandmarks(img_path,size):
	img = skimage.io.imread(img_path) #color image
	img = cv2.resize(img,(size,size))
	landmark_image, mask, layered, points = getLandmarks_Mask(img)
	return img, landmark_image, mask, layered, points


"""
	Save and load data
	Splits data into landmarks, resized images, and overlayed resized images
"""
def processImagesCV2(d,reshaped_dir,landmark_dir,overlay_dir):
	names = list(d.keys())
	print("Processing images, saving reshaped images, landmarks, and overlays")
	bar = tqdm(np.arange(len(names)))
	for i in bar:
		n = names[i]
		reshaped_named_dir = reshaped_dir + "/{}".format(n)
		landmark_named_dir = landmark_dir + "/{}".format(n)
		overlay_named_dir = overlay_dir + "/{}".format(n)
		os.mkdir(reshaped_named_dir)
		os.mkdir(landmark_named_dir)
		os.mkdir(overlay_named_dir)
		seqs = d[n]['sequences']
		for seq in seqs: #sequences for person n
			reshaped_seq_path = reshaped_named_dir + "/{}".format(seq)
			landmark_seq_path = landmark_named_dir + "/{}".format(seq)
			overlay_seq_path = overlay_named_dir + "/{}".format(seq)
			os.mkdir(reshaped_seq_path)
			os.mkdir(landmark_seq_path)
			os.mkdir(overlay_seq_path)
			img_paths = seqs[seq]
			for p in img_paths: #jpgs in sequence seq
				img_name = p.split('/')[-1]
				try:
					reshaped_img, landmark_p, landmark_img, overlay_img, points = processLandmarks(p,224)
				except:
					continue
				reshaped_path = reshaped_seq_path + "/{}".format("reshaped_"+img_name)
				landmark_path = landmark_seq_path + "/{}".format("landmark_"+img_name)
				overlay_path = overlay_seq_path + "/{}".format("overlayed_"+img_name)
				
				scipy.misc.imsave(reshaped_path, reshaped_img)
				scipy.misc.imsave(landmark_path, landmark_img)
				scipy.misc.imsave(overlay_path, overlay_img)
		bar.set_description("Person {} ({}) processed!".format(i,n))

def readImg(path,size=224):
	img = skimage.io.imread(path) #color image
	img = cv2.resize(img,(size,size))
	img = torch.from_numpy(img)
	return img.permute(2,0,1)

def readImages(reshaped_sequence_count,reshaped_dir,landmark_dir,size=224):
	landmark_dict = {}
	reshaped_dict = {}
	landmark_keys = list(landmark_dir.keys())
	reshaped_keys = list(reshaped_dir.keys())
	num_sequences = len(landmark_keys)
	print("Reading in landmark images...")
	landmark_bar = tqdm(np.arange(len(landmark_keys)))
	for i in landmark_bar:
		key = landmark_keys[i]
		sequence = landmark_dir[key]
		landmark_dict[key] = []
		for s in sequence:
			landmark_dict[key].append(readImg(s,size))
		
	print("Reading in reshaped images...")
	reshaped_bar = tqdm(np.arange(len(reshaped_keys)))
	for i in reshaped_bar:
		key = reshaped_keys[i]
		sequence = reshaped_dir[key]
		reshaped_dict[key] = []
		for s in sequence:
			reshaped_dict[key].append(readImg(s,size))

	return reshaped_sequence_count, reshaped_dict, landmark_dict , num_sequences

def deleteSubdirs(directory):
	subs = glob.glob(directory + "/*")
	for s in subs:
		shutil.rmtree(s)
		
def cleanImgs(reshaped_dir,landmark_dir,overlay_dir):
	deleteSubdirs(reshaped_dir)
	deleteSubdirs(landmark_dir)
	deleteSubdirs(overlay_dir)
	
class Saver:
	def __init__(self,checkpoint_path):
		self.checkpoint_path = "checkpoints/" + checkpoint_path

	def makeDir(self):
		try:   
			os.mkdir(self.checkpoint_path)
			print("Checkpoint path successfully created!")
		except:
			print("Checkpoint path exists")

	def saveCheckpoint(self,epoch,args):
		filename = "{}_epoch".format(epoch)
		torch.save(args,self.checkpoint_path + "/{}.tar".format(filename))

class Loader:
	def __init__(self,tar):
		self.tar = tar 

	def loadModel(self):
		checkpoint = torch.load(self.tar)
		return checkpoint