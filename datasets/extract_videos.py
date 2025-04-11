#!/usr/bin/env python3

import os
import skvideo.io
import concurrent.futures
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Extract frames from videos.')
parser.add_argument('-d', '--data_path', help='location of input video folder.', type=str, 
                    choices=['ucf101', 'hmdb51'])
parser.add_argument('--split', help='split of dataset', type=int,
					choices=[1, 2, 3], default=1)
args = parser.parse_args()

ROOT = './'
video_path = os.path.join(ROOT, args.data_path)
frame_path = os.path.join(ROOT, args.data_path +'_frame')
os.makedirs(frame_path, exist_ok=True)

# input
if args.data_path == 'ucf101':
	label_file_path = os.path.join(ROOT, 'ucfTrainTestlist')
	class_file = os.path.join(label_file_path, 'classInd.txt')
	train_file = os.path.join(label_file_path, 'trainlist{:02d}.txt'.format(args.split))
	val_file = os.path.join(label_file_path, 'testlist{:02d}.txt'.format(args.split))
elif args.data_path == 'hmdb51':
	label_file_path = os.path.join(ROOT, 'hmdb51_labels')
	class_file = os.path.join(label_file_path, 'hmdb_labels.txt')
	train_file = os.path.join(label_file_path, 'hmdb51_split{:1d}_train.txt'.format(args.split))
	val_file = os.path.join(label_file_path, 'hmdb51_split{:1d}_test.txt'.format(args.split))

# output
train_img_folder = frame_path
val_img_folder = frame_path
train_list = os.path.join(frame_path, 'train.txt')
val_list = os.path.join(frame_path, 'val.txt')


def load_categories(file_path):
    id_to_label = {}
    label_to_id = {}
    with open(file_path) as f:
        cls_id = 0
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            label_id, label = line.split()
            id_to_label[cls_id] = label
            label_to_id[label] = cls_id
            cls_id += 1
    return id_to_label, label_to_id

_, label_to_id = load_categories(class_file)


def load_video_list(file_path, test=False):
    videos = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            if not test:
                video, label_id = line.split()
            else:
                video = line
            label_name, vname = video.split('/')
            videos.append([vname.split('.')[0], label_name])
            # videos.append([video.split('.')[0], label_name])
    return videos


def resize_to_short_side(h, w, short_side=360):
    newh, neww = h, w
    if h < w:
        newh = short_side
        neww = (w / h) * newh
    else:
        neww = short_side
        newh = (h / w) * neww
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return newh, neww


def video_to_images(video, basedir, targetdir, short_side=256):
    try:
        cls_id = label_to_id[video[1]]
    except:
        cls_id = -1
    filename = os.path.join(basedir, video[0] + ".avi")
    output_foldername = os.path.join(targetdir, video[0])
    if not os.path.exists(filename):
        print("{} does not exist.".format(filename))
        return video[0], cls_id, 0
    else:
        try:
            video_meta = skvideo.io.ffprobe(filename)
            height = int(video_meta['video']['@height'])
            width = int(video_meta['video']['@width'])
        except:
            print("Can not get video info: {}".format(filename))
            return video[0], cls_id, 0

        if width > height:
            scale = "scale=-1:{}".format(short_side)
        else:
            scale = "scale={}:-1".format(short_side)
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)

        command = ['ffmpeg',
                   '-i', '"%s"' % filename,
                   '-vf', scale,
                   '-threads', '1',
                   '-loglevel', 'panic',
                   '-q:v', '0',
                   '{}/'.format(output_foldername) + '"%05d.jpg"']
        command = ' '.join(command)
        try:
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except:
            print("fail to convert {}".format(filename))
            return video[0], cls_id, 0

        # get frame num
        i = 0
        while True:
            img_name = os.path.join(output_foldername, '{:05d}.jpg'.format(i + 1))
            if os.path.exists(img_name):
                i += 1
            else:
                break

        frame_num = i
        print("Finish {}, id: {} frames: {}".format(filename, cls_id, frame_num))
        return video[0], cls_id, frame_num


def create_train_video(short_side):
    with open(train_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_path, train_img_folder, int(short_side))
                   for video in train_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {} {}".format(video_id, frame_num, label_id), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos), flush=True)
            curr_idx += 1
    print("Completed")


def create_val_video(short_side):
    with open(val_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_path, val_img_folder, int(short_side))
                   for video in val_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {} {}".format(video_id, frame_num, label_id), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos))
            curr_idx += 1
    print("Completed")


if __name__ == '__main__':
    train_videos = load_video_list(train_file)
    val_videos = load_video_list(val_file, True)

    create_train_video(256)
    create_val_video(256)