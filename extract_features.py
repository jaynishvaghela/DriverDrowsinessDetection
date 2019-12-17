import numpy as np
import os.path
import csv
import glob
import tensorflow as tf
import h5py as h5py
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
import natsort
import sys

seq = int(sys.argv[1])


def extractor(image_path):

	with open('./output_graph.pb', 'rb') as graph_file:
		graph_def = tf.compat.v1.GraphDef()
		graph_def.ParseFromString(graph_file.read())
		tf.import_graph_def(graph_def, name='')

	with tf.compat.v1.Session() as sess:
	    pooling_tensor = sess.graph.get_tensor_by_name('pool_3:0')
	    #tf.compat.v2.io.gfile.GFile()
	    image_data = tf.compat.v1.gfile.FastGFile(image_path, 'rb').read()	
	    pooling_features = sess.run(pooling_tensor, \
	    	{'DecodeJpeg/contents:0': image_data})
	    pooling_features = pooling_features[0]

	return pooling_features

def extract_features():
	with open('data/data_file.csv','r') as f:
		reader = csv.reader(f)
		for videos in reader:
			path = os.path.join('data', 'sequences', videos[0], videos[2] + '-' + str(seq) + '-features')
			print(path)
			path_frames = os.path.join('data', videos[0], videos[1])
			print(path_frames)
			filename = videos[2]
			#print(filename)
			frames = glob.glob(os.path.join(path_frames, filename + '/*jpg'))
			frames  = natsort.natsorted(frames,reverse=False)
			sequence = []
			cnt = 0
			for image in frames:
				with tf.Graph().as_default():
					features = extractor(image)
					cnt+=1
					print('Appending sequence of image:',image,' of the video:',videos)
					sequence.append(features)

				if cnt % seq == 0:
					np.save(path+str(cnt)+'.npy',sequence)
					sequence = []
			print('Sequences saved successfully')

extract_features()																																																																				
