import FaceLandmark as fl
import os
import cv2

dataset_path = './dataset/'
#dataset_label = ['A/','C/','D/','J/','N/','P/','S/','T/']
dataset_label = ['S/']
Detector = fl.FaceLandmark()
for label in dataset_label:
    vids = os.listdir(dataset_path+label)
    for vid in vids:
        vid_path = dataset_path+label+vid
        print("Now working on: %s"%vid_path)
        cap = cv2.VideoCapture(vid_path)
        ret,frame = cap.read()
        Detector.change_vid(vid_path)
        Detector.mark_face()
        #Detector.draw_landmark()
        #Detector.save_marked_frames(tag=vid)
        Detector.save_lip(tag=vid[:-4])

        

