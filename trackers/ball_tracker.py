from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
	def __init__(self, model_path):
		self.model = YOLO(model_path)

	# function to predict position of balls whose position was not detected in video by model
	def interpolate_ball_positions(self, ball_positions):
		ball_positions = [x.get(1, []) for x in ball_positions] # the list will be empty when there are no detections
		df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

		# interpolate the missing values
		df_ball_positions = df_ball_positions.interpolate()
		df_ball_positions = df_ball_positions.bfill() # prevents interpolating first frame

		ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

		return ball_positions

	def get_ball_shot_frames(self, ball_positions):
		ball_positions = [x.get(1, []) for x in ball_positions] # the list will be empty when there are no detections
		df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

		df_ball_positions['ball_hit'] = 0

		# get center coordinate of ball
		df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
		df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window = 5, min_periods = 1, center = False).mean() # remove outliers
		# identify when the y changes
		df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff() # calculates diff btwn consecutive rows
		minimum_change_frames_for_hit = 25 # approx number of frames for which there was no abrupt change in ball direction
		for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)): # 1.2 = increase by 20%
		    negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0 # first increasing and then decreasing
		    positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0 # first decreasing and then increasing

		    if negative_position_change or positive_position_change:
		        change_count = 0 # keeps track of how many frames the change goes on for
		        for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
		            negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0 # compare original frame with future frames
		            positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

		            if negative_position_change and negative_position_change_following_frame:
		                change_count += 1
		            elif positive_position_change and positive_position_change_following_frame:
		                change_count += 1

		        if change_count>minimum_change_frames_for_hit-1:
		            df_ball_positions['ball_hit'].iloc[i] = 1 # If enough frames (≥ 25) show the direction has changed and stayed changed, it flags the frame i as a ball hit.
		frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

		return frame_nums_with_ball_hits

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