wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
mkdir ucf101 && unrar e UCF101.rar ucf101
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zi --no-check-certificate && unzip UCF101TrainTestSplits-RecognitionTask.zip
python extract_videos.py -d ucf101
python create_lmdb.py -d ucf101_frame -s train -vr 0 9437
python create_lmdb.py -d ucf101_frame -s val -vr 0 3755