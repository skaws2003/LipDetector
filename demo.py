import FaceLandmark as fl
import cv2

vid_path = './example.avi'

Detector = fl.FaceLandmark()
Detector.change_vid(vid_path)
Detector.mark_face()
Detector.draw_landmark()
Detector.save_marked_frames()