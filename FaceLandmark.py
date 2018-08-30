# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
 

ERROR_NO_FACE = -1
ERROR_MANY_FACE = -2
RESULT_PATH = './result/'
LOG_PATH = 'log.txt'
LIP_EMP=0.3


def shape_to_list(shape):
	coords = []
	for i in range(0, 68):
		coords.append((shape.part(i).x, shape.part(i).y))
	return coords


class FaceLandmark:
		
	def __init__(self):
		# Buffers
		self.frame_buffer = []	# Saves grascaled frames
		self.frame_buffer_color = []
		self.marked_buffer = []	# Saves the detected landmark array (frame#,landmark list)
		self.log_buffer = []	# Saves any error logs	(frame#,errorcode)
		self.marked_frames = []	# The images marked with landmarks
		self.validity = []	# Saves the validity of the frame

		self.is_valid = False	# Check if every frames marked correctly
		# Set up predictor
		self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		self.detector = dlib.get_frontal_face_detector()

		# Clear log file
		logfile = open(LOG_PATH,'w')
		logfile.close()

	def write_log_file(self):
		"""
		Writes log file.
		"""
		print("Writing log file...")
		logfile = open('log.txt','w')
		for (code,path) in self.log_buffer:
			if code == ERROR_MANY_FACE:
				logfile.write("TOO MANY FACE: %s"%path)
			elif code == ERROR_NO_FACE:
				logfile.write("NO FACE: %s"%path)
		logfile.close()

	def change_vid(self,vid_path):
		"""
		change the video working on

		args
		 - vid_path: path of the video file
		"""
		# Open video
		self.vid_path = vid_path
		self.vid = cv2.VideoCapture(vid_path)
		if not self.vid.isOpened():
			print("Video file does not exist")
			return
		self.build_buffer()

	def build_buffer(self):
		"""
		Builds video frame buffer.
		Only for inner use.
		"""
		self.frame_buffer = []
		self.frame_buffer_color = []
		while(True):
			success, frame = self.vid.read()
			if not success: break
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			self.frame_buffer.append(gray)
			self.frame_buffer_color.append(frame)
		self.vid.release()
	
	def mark_face(self):
		"""
		put marker information in self.marked_buffer
		If error occurs, add it to the log file.
		return : validity of the marked video
		"""
		self.marked_buffer = []
		self.is_valid = True
		# Mark image with facial landmark
		for (i, image) in enumerate(self.frame_buffer):
			# Recognize faces
			face_rects = self.detector(image,1)
			# Choose a face if more than one
			if len(face_rects) < 1:
				print("no face detected! frame no.%d"%i)
				self.log_buffer.append((i,ERROR_NO_FACE))
				logfile = open(LOG_PATH,'a')
				logfile.write(self.vid_path + " : No face detected\n")
				logfile.close()
				self.marked_buffer.append((i,None))
				self.is_valid = False
				break
			elif len(face_rects) > 1:
				print("face more than one! frame no.%d"%i)
				self.log_buffer.append((i,ERROR_MANY_FACE))
				logfile = open(LOG_PATH,'a')
				logfile.write(self.vid_path + " : Too many face detected\n")
				logfile.close()
				self.marked_buffer.append((i,None))
				self.is_valid = False
				break
			# Determine facial landmarks and change it into list format
			rect = face_rects[0]
			shape = self.predictor(image, rect)
			shape = shape_to_list(shape)
			self.marked_buffer.append((i,shape))
		#self.write_log_file()
		return self.is_valid

	def get_frames(self):
		"""
		Return frame buffer
		"""
		return self.frame_buffer

	def draw_landmark(self,force=False):
		"""
		Draw the landmarks on self.marked_frames

		Fields:
		 - force: even if the video is not valid, just leave the invalid frame.
		"""
		if not self.is_valid:
			print("The video is not valid! will not draw landmark")
			return False
		
		print("Drawing landmarks on the frame buffer...")
		for (i,landmark) in self.marked_buffer:
			img = self.frame_buffer[i]
			for (x,y) in landmark:
				
				img = cv2.circle(img,(x,y),1,(0,0,255),-1)
			self.marked_frames.append(img)
				

	def save_marked_frames(self,tag="Result"):
		"""
		Saves the resulting marked frames
		"""
		if not self.is_valid:
			print("The video is not valid! will not save images")
			return False
		print("Saving frame buffer...")
		result_path = RESULT_PATH+tag
		for (i,frame) in enumerate(self.marked_frames):
			cv2.imwrite(result_path+"_%d"%i+".bmp",frame)
		print("done!")

	def save_frames(self,tag="Result"):
		"""
		Saves the resulting marked frames
		"""
		if not self.is_valid:
			print("The video is not valid! will not save images")
			return False
		print("Saving frame buffer...")
		result_path = RESULT_PATH+tag
		for (i,frame) in enumerate(self.frame_buffer):
			cv2.imwrite(result_path+"_%d"%i+".bmp",frame)
		print("done!")

	def save_lip(self,tag="Result"):
		"""
		Saves lip image frames
		"""
		if not self.is_valid:
			print("The video is not valid! will not save lip images")
			return False
		print("Saving lip images...")
		result_path = RESULT_PATH+tag

		cropped_buffer = []
		for (i,arr) in self.marked_buffer:
			print(len(arr))
			lip_arr = arr[48:68]
			lip_arr_x = sorted(lip_arr,key = lambda pointx: pointx[0])
			lip_arr_y = sorted(lip_arr, key = lambda pointy: pointy[1])
			x_add = int((-lip_arr_x[0][0]+lip_arr_x[-1][0])*LIP_EMP)
			y_add = int((-lip_arr_y[0][1]+lip_arr_y[-1][1])*LIP_EMP)
			crop_pos = (lip_arr_x[0][0]-x_add, lip_arr_x[-1][0]+x_add, lip_arr_y[0][1]-y_add, lip_arr_y[-1][1]+y_add)
			cropped = self.frame_buffer_color[i][crop_pos[2]:crop_pos[3],crop_pos[0]:crop_pos[1]]
			cropped = cv2.resize(cropped,(64,64),interpolation=cv2.INTER_CUBIC)
			cropped_buffer.append(cropped)

			directory = RESULT_PATH + tag + "/"
		for (i,frame) in enumerate(cropped_buffer):
			if not os.path.exists(directory):
				os.makedirs(directory)
			cv2.imwrite(directory + "%d"%(i+1) + ".jpg", frame)
		print("Saving lip done.")