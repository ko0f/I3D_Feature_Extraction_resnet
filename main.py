from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
from extract_features import run
from resnet import i3_res50
from tqdm import tqdm
import os.path

def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode, use_cuda, vid_ext, overwrite):
	Path(outputpath).mkdir(parents=True, exist_ok=True)
	temppath = outputpath + "/temp/"
	if os.path.isdir(temppath):
		shutil.rmtree(temppath) # make sure temp dir is empty, can be leftovers
	rootdir = Path(datasetpath)
	videos = [str(f) for f in rootdir.glob('**/*.' + vid_ext)]
	# setup the model
	i3d = i3_res50(400, pretrainedpath)
	if use_cuda:
		i3d.cuda()
	i3d.train(False)  # Set model to evaluate mode
	for video in tqdm(videos):
		videoname = video.split("/")[-1].split(".")[0]
		features_output_fn = outputpath + "/" + videoname
		if not overwrite and os.path.isfile(features_output_fn + '.npy'):
			continue # skip extracted features
		startime = time.time()
		print("Generating JPG files for each frame of {0}...".format(video))
		Path(temppath).mkdir(parents=True, exist_ok=True)
		ffmpeg \
			.input(video) \
			.filter('fps', fps=16, round='up') \
			.output('{}%d.jpg'.format(temppath),start_number=0) \
			.global_args('-loglevel', 'quiet') \
			.run()
		print("Extracting features from frames...")
		features = run(i3d, frequency, temppath, batch_size, sample_mode, use_cuda)
		np.save(features_output_fn, features)
		print("Obtained features of shape", features.shape, "processing time {0} seconds".format(time.time() - startime))
		shutil.rmtree(temppath)

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str, default="samplevideos/")
	parser.add_argument('--outputpath', type=str, default="output")
	parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=20)
	parser.add_argument('--sample_mode', type=str, default="oversample")
	parser.add_argument('--ext', type=str, default="mkv")
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--overwrite', action='store_true')
	args = parser.parse_args()
	generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode, args.cuda, args.ext, args.overwrite)    
