import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm


def load_frame(frame_file):
	''' loads a frame and normalizes RGB values '''
	data = Image.open(frame_file)
	# sample down frame to 340x256 resolution
	data = data.resize((340, 256), Image.LANCZOS)
	data = np.array(data)
	data = data.astype(float)
	# rescale RGB values from 0 to 255 info -1 to 1
	data = (data * 2 / 255) - 1
	assert(data.max()<=1.0)
	assert(data.min()>=-1.0)
	return data


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
	''' fills a batch with frame data, RGB values normalized into -1 to 1 '''
	batch_data = np.zeros(frame_indices.shape + (256,340,3))
	for i in range(frame_indices.shape[0]): # batch_size (20)
		for j in range(frame_indices.shape[1]): # chunk_size (16)
			frame_fn = os.path.join(frames_dir, rgb_files[frame_indices[i][j]])
			batch_data[i,j,:,:,:] = load_frame(frame_fn)
	return batch_data


def oversample_data(data):
	''' arranges a frame RGB data into 10crop format '''
	data_flip = np.array(data[:,:,:,::-1,:])

	data_1 = np.array(data[:, :, :224, :224, :])
	data_2 = np.array(data[:, :, :224, -224:, :])
	data_3 = np.array(data[:, :, 16:240, 58:282, :])
	data_4 = np.array(data[:, :, -224:, :224, :])
	data_5 = np.array(data[:, :, -224:, -224:, :])

	data_f_1 = np.array(data_flip[:, :, :224, :224, :])
	data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
	data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
	data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
	data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

	return [data_1, data_2, data_3, data_4, data_5,
		data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]

def forward_batch(i3d, b_data, use_cuda):
	b_data = b_data.transpose([0, 4, 1, 2, 3])
	b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224
	with torch.no_grad():
		if use_cuda:
			b_data = b_data.cuda()
		b_data = Variable(b_data).float()
		inp = {'frames': b_data}
		features = i3d(inp)
	return features.cpu().numpy()


def run(i3d, chunk_size, frames_dir, batch_size, sample_mode, use_cuda):
	''' extracts feature set from given dir with frames JPGs of one video '''
	
	assert(sample_mode in ['oversample', 'center_crop'])
	rgb_files = natsorted([i for i in os.listdir(frames_dir)])
	frame_cnt = len(rgb_files)
	# Cut frames
	assert(frame_cnt > chunk_size)
	clipped_length = frame_cnt - chunk_size
	clipped_length = (clipped_length // chunk_size) * chunk_size  # The start of last chunk
	frame_indices = [] 
	# frame_indices (chunk_num, chunk_size): list of 16 frame indices 
	# (300, 16) for 5 min footage
	for i in range(clipped_length // chunk_size + 1):
		frame_indices.append([j for j in range(i * chunk_size, i * chunk_size + chunk_size)])
	frame_indices = np.array(frame_indices)
	chunk_num = frame_indices.shape[0]
	batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
	# we'll send batch_size number of chunks to the model each time
	frame_indices = np.array_split(frame_indices, batch_num, axis=0)
	# frame_indices shape (batch_num, batch_size, chunk_size) = (15, 20, 16) for 5 min footage
	
	if sample_mode == 'oversample':
		full_features = [[] for i in range(10)]
	else:
		full_features = [[]]

	print('Processing frames through I3D+10crop in batches...')
	for batch_id in tqdm(range(batch_num)): 
		batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id]) # load batch frames
		if(sample_mode == 'oversample'):
			batch_data_ten_crop = oversample_data(batch_data)
			for i in range(10):
				assert(batch_data_ten_crop[i].shape[-2]==224)
				assert(batch_data_ten_crop[i].shape[-3]==224)
				temp = forward_batch(i3d, batch_data_ten_crop[i], use_cuda)
				full_features[i].append(temp)

		elif(sample_mode == 'center_crop'):
			batch_data = batch_data[:,:,16:240,58:282,:]
			assert(batch_data.shape[-2]==224)
			assert(batch_data.shape[-3]==224)
			temp = forward_batch(i3d, batch_data, use_cuda)
			full_features[0].append(temp)
	
	full_features = [np.concatenate(i, axis=0) for i in full_features]
	full_features = [np.expand_dims(i, axis=0) for i in full_features]
	full_features = np.concatenate(full_features, axis=0)
	full_features = full_features[:,:,:,0,0,0]
	full_features = np.array(full_features).transpose([1,0,2])
	return full_features
