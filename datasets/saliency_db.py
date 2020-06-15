import torch
import torch.utils.data as data
from PIL import Image
import os
import functools
import copy
import numpy as np
from numpy import median
import scipy.io as sio
import torchaudio


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			img = img.convert('RGB')
			new_size = (320,240)
			img = img.resize(new_size)
			return img

def pil_loader_sal(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			img = img.convert('L')
			new_size = (320,240)
			img = img.resize(new_size)
			return img

def accimage_loader(path):
	try:
		import accimage
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)

def get_default_image_loader():
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader
	else:
		return pil_loader

def video_loader(video_dir_path, frame_indices, image_loader):
	video = []
	for i in frame_indices:
		image_path = os.path.join(video_dir_path, 'img_{:05d}.jpg'.format(i))
		if os.path.exists(image_path):
			video.append(image_loader(image_path))
		else:
			return video
	return video

def get_default_video_loader():
	image_loader = get_default_image_loader()
	return functools.partial(video_loader, image_loader=image_loader)

def read_sal_text(txt_file):
	test_list = {'names': [], 'nframes': [], 'fps': []}
	with open(txt_file,'r') as f:
		for line in f:
			word=line.split()
			test_list['names'].append(word[0])
			test_list['nframes'].append(word[1])
			test_list['fps'].append(word[2])
	return test_list


def make_dataset(root_path, annotation_path, salmap_path, audio_path,
				 step, step_duration):
	data = read_sal_text(annotation_path)
	video_names = data['names']
	video_nframes = data['nframes']
	video_fps = data['fps']
	dataset = []
	audiodata= []
	for i in range(len(video_names)):
		if i % 100 == 0:
			print('dataset loading [{}/{}]'.format(i, len(video_names)))

		video_path = os.path.join(root_path, video_names[i])
		annot_path = os.path.join(salmap_path, video_names[i], 'maps')
		annot_path_bin = os.path.join(salmap_path, video_names[i])
		if not os.path.exists(video_path):
			continue
		if not os.path.exists(annot_path):
			continue
		if not os.path.exists(annot_path_bin):
			continue

		n_frames = int(video_nframes[i])
		if n_frames <= 1:
			continue

		begin_t = 1
		end_t = n_frames

		audio_wav_path = os.path.join(audio_path,video_names[i],video_names[i]+'.wav')
		if not os.path.exists(audio_wav_path):
			continue
		[audiowav,Fs]=torchaudio.load(audio_wav_path, normalization=False)
		audiowav = audiowav * (2 ** -23)
		n_samples = Fs/float(video_fps[i])
		starts=np.zeros(n_frames+1, dtype=int)
		ends=np.zeros(n_frames+1, dtype=int)
		starts[0]=0
		ends[0]=0
		for videoframe in range(1,n_frames+1):
			startemp=max(0,((videoframe-1)*(1.0/float(video_fps[i]))*Fs)-n_samples/2)
			starts[videoframe] = int(startemp)
			endtemp=min(audiowav.shape[1],abs(((videoframe-1)*(1.0/float(video_fps[i]))*Fs)+n_samples/2))
			ends[videoframe] = int(endtemp)

		audioinfo = {
			'audiopath': audio_path,
			'video_id': video_names[i],
			'Fs' : Fs,
			'wav' : audiowav,
			'starts': starts,
			'ends' : ends
		}
		audiodata.append(audioinfo)

		sample = {
			'video': video_path,
			'segment': [begin_t, end_t],
			'n_frames': n_frames,
			'fps': video_fps[i],
			'video_id': video_names[i],
			'salmap': annot_path,
			'binmap': annot_path_bin
		}
		step=int(step)
		for j in range(1, n_frames, step):
			sample_j = copy.deepcopy(sample)
			sample_j['frame_indices'] = list(range(j, min(n_frames + 1, j + step_duration)))
			dataset.append(sample_j)

	return dataset, audiodata


class saliency_db(data.Dataset):

	def __init__(self,
				 root_path,
				 annotation_path,
				 subset,
				 audio_path,
				 spatial_transform = None,
				 temporal_transform = None,
				 target_transform = None,
				 exhaustive_sampling = False,
				 sample_duration = 16,
				 step_duration = 90,
				 get_loader = get_default_video_loader):

		if exhaustive_sampling:
			self.exhaustive_sampling = True
			step = 1
			step_duration = sample_duration
		else:
			self.exhaustive_sampling = False
			step = max(1, step_duration - sample_duration)

		self.data,self.audiodata = make_dataset(
			root_path, annotation_path, subset, audio_path,
			step, step_duration)

		self.spatial_transform = spatial_transform
		self.temporal_transform = temporal_transform
		self.target_transform = target_transform
		self.loader = get_loader()
		max_audio_Fs = 22050
		min_video_fps = 10
		self.max_audio_win = int(max_audio_Fs / min_video_fps * sample_duration)

	def __getitem__(self, index):

		path = self.data[index]['video']
		annot_path = self.data[index]['salmap']
		annot_path_bin = self.data[index]['binmap']

		frame_indices = self.data[index]['frame_indices']
		if self.temporal_transform is not None:
			frame_indices = self.temporal_transform(frame_indices)

		video_name=self.data[index]['video_id']
		flagexists=0
		for iaudio in range(0, len(self.audiodata)-1):
			if (video_name == self.audiodata[iaudio]['video_id']):
				audioind = iaudio
				flagexists = 1
				break

		audioexcer  = torch.zeros(1,self.max_audio_win)  ## maximum audio excerpt duration
		data = {'rgb':[], 'audio':[]}
		valid = {}
		valid['audio']=0
		if flagexists:
			frame_ind_start = frame_indices[0]
			frame_ind_end = frame_indices[len(frame_indices)-1]
			excerptstart = self.audiodata[audioind]['starts'][frame_ind_start]
			excerptend = self.audiodata[audioind]['ends'][frame_ind_end]
			try:
				valid['audio'] = self.audiodata[audioind]['wav'][:, excerptstart:excerptend+1].shape[1]
			except:
				pass
			audioexcer_tmp = self.audiodata[audioind]['wav'][:, excerptstart:excerptend+1]
			if (valid['audio']%2)==0:
				audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2))] = \
					torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
			else:
				audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2)+1)] = \
					torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
			data['audio'] = audioexcer.view(1,1,-1)
		else:
			data['audio'] = audioexcer.view(1,1,-1)

		med_indices = int(round(median(frame_indices)))
		target = {'salmap':[],'binmap':[]}
		target['salmap'] = pil_loader_sal(os.path.join(annot_path, 'eyeMap_{:05d}.jpg'.format(med_indices)))
		tmp_mat = sio.loadmat(os.path.join(annot_path_bin, 'fixMap_{:05d}.mat'.format(med_indices)))
		binmap_np = np.array(Image.fromarray(tmp_mat['eyeMap'].astype(float)).resize((320, 240), resample = Image.BILINEAR)) > 0
		target['binmap'] = Image.fromarray((255*binmap_np).astype('uint8'))
		if self.exhaustive_sampling:
			target['video'] = self.data[index]['video_id']
		clip = self.loader(path, frame_indices)

		if self.spatial_transform is not None:
			self.spatial_transform.randomize_parameters()
			self.spatial_transform_sal = copy.deepcopy(self.spatial_transform)
			del self.spatial_transform_sal.transforms[-1]
			clip = [self.spatial_transform(img) for img in clip]
			target['salmap'] = self.spatial_transform_sal(target['salmap'])
			target['binmap'] = self.spatial_transform_sal(target['binmap'])
			target['binmap'] = torch.gt(target['binmap'], 0.0).float()
		try:
			clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
		except:
			pass

		valid['sal'] = 1
		data['rgb'] = clip

		return data, target, valid

	def __len__(self):
		return len(self.data)
