#Input video name format:"test<Alert/Drowsy><test subject number>"
#Output video name format:"test<Alert/Drowsy><test subject number>"

#-------------------------------------------------------------------x

# Face Detection

#-------------------------------------------------------------------
hyper=1000
seq=50
#train_test=0.8
# i=1
# if [ -d "Output" ] 
# then 
#   rm -Rf Output
# fi
# mkdir Output
# while [ $i -le 6 ]
# do
#   #extract patches for alert subjects
#   mkdir Output/$i
#   mkdir Output/$i/Alert
#   echo "Alert $i"
#   python face_detection.py $i Input/$i/Alert_Video$i.mp4 ./Output/$i/Alert/ $hyper
#   i=$(($i + 1))
# done

# i=1
# while [ $i -le 6 ]
# do
#   #extract patches for drowsy subjects
#   mkdir Output/$i/Drowsy
#   echo "Drowsy $i"
#   python face_detection.py $i Input/$i/Drowsy_Video$i.mp4 ./Output/$i/Drowsy/ $hyper
#   i=$(($i + 1))
# done

# # #-------------------------------------------------------------------x

# # # Transfer Learning on Inception V3

# #-------------------------------------------------------------------
# if [ -d "training" ] 
# then 
#   rm -Rf trianing
# fi
# if [ -d "testing" ] 
# then 
#   rm -Rf testing
# fi
# if [ -d "sequences" ] 
# then 
#   rm -Rf sequences
# fi

# python csv_file_creator.py $train_test
# python retrain.py --output_graph=./output_graph.pb --output_labels=./output_labels.txt --image_dir=./data/training/

# #-------------------------------------------------------------------x

# # Extract Features

# #-------------------------------------------------------------------
# python extract_features.py $seq
#-------------------------------------------------------------------x

# Training 

#-------------------------------------------------------------------
#extract patches for alert subjects
python train_test.py $seq $hyper
