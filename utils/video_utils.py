import cv2

# return frames of the video
def read_video(video_path):
	cap = cv2.VideoCapture(video_path)
	frames = []
	while True:
		ret, frame = cap.read()
		if not ret: # ret returns false if no more frames are present in video
			break
		frames.append(frame)
	cap.release()
	return frames

# function to save the processed video
def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    # 24 = 24fps
    for frame in output_video_frames:
        out.write(frame)
    out.release()