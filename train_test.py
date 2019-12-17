from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import Dataset
import time
import os
import os.path
import numpy as np
import sys
import pandas as pd
import csv


seq=int(sys.argv[1])
hyper = int(sys.argv[2])
df = pd.read_csv("./data/data_file.csv", header = None)
#df_res = pd.read_csv("./data/result_file.csv", header = None)
df_res = pd.DataFrame()
	#columns=['train_test', 'Video', 'Accuracy', 'Prediction'])
batch_size = 4
nb_epoch = 10
test_no = df[df.iloc[:,0] == 'testing'].shape[0]
train_no = df[df.iloc[:,0] == 'training'].shape[0]
#test_no = df[df.iloc[:,0] == 'testing'].shape[0]
#def train(batch_size=4, nb_epoch=10):

checkpointer = ModelCheckpoint(filepath=os.path.join('data', 'checkpoints', 'lstm' + '-' + 'features' + '.{epoch:03d}-{val_loss:.3f}.hdf5'),verbose=1,save_best_only=True)


tb = TensorBoard(log_dir=os.path.join('data', 'logs', 'lstm'))

early_stopper = EarlyStopping(patience=5)

timestamp = time.time()

csv_logger = CSVLogger(os.path.join('data', 'logs', 'lstm' + '-' + 'training-' + str(timestamp) + '.log'))

data = Dataset(
        seq_length=seq,
        class_limit=2,
    )

steps_per_epoch = 4

X, y = data.get_all_sequences_in_memory('training', hyper, seq)
X_test, y_test = data.get_all_sequences_in_memory('testing', hyper, seq)

#X_test, y_test = data.get_all_sequences_in_memory('testing', cnt, seq)

rm = ResearchModels(len(data.classes),'lstm',data.seq_length, None)
print("##################################################")
#X=X[2:]
#X_test=X_test[2:]
print(X.shape)
X=np.ravel(X)
# First argument is number of training videos, second is number of images within it
print(X.shape)
#print(train_no)
#print(train_no * seq)
X=X.reshape((160, seq,-1))
print(X.shape)
#X_test=np.ravel(X_test)
# First argument is number of test videos, second is number of images within it
#X_test=X_test.reshape(test_no,seq,-1)
#print "X", X[0:10]
print("X.shape", X.shape)
print("y.shape", y.shape)
#print("X_test.shape" ,X_test.shape)
#print("y_test.shape" ,y_test.shape)
print("##################################################")

rm.model.fit(X,y,
        batch_size=batch_size,
        validation_data=(X, y),
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger],
		epochs=nb_epoch)
model_json = rm.model.to_json()
with open("rm.model.json",'w') as json_file:
	json_file.write(model_json)
rm.model.save_weights("rm.model.h5")
print("Model saved")
predictions = rm.model.predict(X)
loss, accuracy = rm.model.evaluate(X, y)

# modellstm = rm.model.fit(X,y,
#            batch_size=batch_size,
#            validation_data=(X, y),
#            verbose=1,
#            callbacks=[tb, early_stopper, csv_logger],
# 		epochs=nb_epoch)
# model_json = modellstm.model.to_json()
# with open("modellstm.json",'w') as json_file:
# 	json_file.write(model_json)
# print("Model saved")
# predictions = modellstm.predict(X)
# loss, accuracy = modellstm.evaluate(X, y)

with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
	reader = csv.reader(fin)
	data = list(reader)

filenames = []
for videos in data:
	if(videos[0] == 'training'):
		i = 1
		while i <= int(hyper/seq):
			cnt = i*seq
			filenames.append(videos[2] + '-' + str(seq) + '-' + 'features' + str(cnt)+'.npy')
			i+=1

#print 'Loss:',loss*100,'%'
k = 0
for j in predictions:
	if j[0]>j[1]:
		print("Driver is alert with the confidence of",(j[0]*100),"%")
		df_res = df_res.append({'train_test': 'training', 'Video': filenames[k], 'Accuracy': accuracy, 'Prediction': j[0],'Prediciton_class': 'Alert'},ignore_index = True)
	else:
		print("Driver is drowsy with the confidence of",(j[1]*100),"%")
		df_res = df_res.append({'train_test': 'training', 'Video': filenames[k], 'Accuracy': accuracy, 'Prediction': j[1],'Prediciton_class': 'Drowsy'},ignore_index = True)
		print("Sounding the alarm now....")
		# duration = 10  # second
		# freq = 440  # Hz
		# os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
		#for i in range(5):
		#	os.system('say "Wake up now"')
	k+=1
#df_res.to_csv('./data/result_file.csv',index=False,header=False)


#rm = ResearchModels(len(data.classes),'lstm',data.seq_length, None)
print("##################################################")
#X=X[2:]
#X_test=X_test[2:]
# First argument is number of training videos, second is number of images within it
X_test=np.ravel(X_test)
# First argument is number of test videos, second is number of images within it
X_test=X_test.reshape((40,seq,-1))
#print "X", X[0:10]
print("X_test.shape" ,X_test.shape)
print("y_test.shape" ,y_test.shape)
print("##################################################")
predictions = rm.model.predict(X_test)
loss, accuracy = rm.model.evaluate(X_test, y_test)
with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
	reader = csv.reader(fin)
	data = list(reader)

filenames = []
for videos in data:
	if(videos[0] == 'testing'):
		i = 1
		while i <= int(hyper/seq):
			cnt = i*seq
			filenames.append(videos[2] + '-' + str(seq) + '-' + 'features' + str(cnt)+'.npy')
			i+=1
#print 'Loss:',loss*100,'%'
k = 0
for j in predictions:
	if j[0]>j[1]:
		print("Driver is alert with the confidence of",(j[0]*100),"%")
		df_res = df_res.append({'train_test': 'testing', 'Video': filenames[k], 'Accuracy': accuracy, 'Prediction': j[0],'Prediciton_class': 'Alert'},ignore_index = True)
		df_res = df_res.append({'train_test': 'testing', 'Video': filenames[k], 'Accuracy': accuracy, 'Prediction': j[0],'Prediciton_class': 'Alert'},ignore_index = True)
	else:
		print("Driver is drowsy with the confidence of",(j[1]*100),"%")
		df_res = df_res.append({'train_test': 'testing', 'Video': filenames[k], 'Accuracy': accuracy, 'Prediction': j[1],'Prediciton_class': 'Drowsy'},ignore_index = True)
		print("Sounding the alarm now....")
		# duration = 10  # second
		# freq = 440  # Hz
		# os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
		# for i in range(5):
		# 	os.system('say "Wake up now"')
	k+=1
df_res.to_csv('./data/result_file.csv',index=False,header=False)
			
			
# def main():

# 	# model can be one of lstm, lrcn, mlp, conv_3d, c3d
# 	    # Chose images or features and image shape based on network.

# 	    train(batch_size=batch_size, nb_epoch=nb_epoch)
