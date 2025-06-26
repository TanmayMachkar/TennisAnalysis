from ultralytics import YOLO
import cv2
import pickle

class BallTracker:
	def __init__(self, model_path):
		self.model = YOLO(model_path)

	# detect multiple frames
	def detect_frames(self, frames, read_from_stub = False, stub_path = None): # read_from_stub is used to avoid running model again and again, if True just read stored model from stub_path
		ball_detections = []

		if read_from_stub and stub_path is not None:
			with open(stub_path, 'rb') as f:
				ball_detections = pickle.load(f)
			return ball_detections

		for frame in frames:
			ball_dict = self.detect_frame(frame)
			ball_detections.append(ball_dict)

		if stub_path is not None:
			with open(stub_path, 'wb') as f:
				pickle.dump(ball_detections, f)

		return ball_detections

	# detect single frame
	def detect_frame(self, frame):
		results = self.model.predict(frame, conf = 0.15)[0]
		# persist = True since the video is not given in frames and not at once so model needs to remember the prev tracks as well
		
		ball_dict = {}
		for box in results.boxes:
			result = box.xyxy.tolist()[0] # xmax, ymax, xmin, ymin
			ball_dict[1] = result

		return ball_dict

	# function to draw the bounding boxes
	def draw_bboxes(self, video_frames, ball_detections):
		output_video_frames = []
		for frame, ball_dict in zip(video_frames, ball_detections):
			# Draw bounding box
			for track_id, bbox in ball_dict.items():
				x1, y1, x2, y2 = bbox
				cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]),int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2) # bbox[0] and bbox[1] are the min x and y
				cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2) # 2 = only borders are filled the entire reactangles are not
			output_video_frames.append(frame)

		return output_video_frames