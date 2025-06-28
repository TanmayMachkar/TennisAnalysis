import cv2
import sys
import numpy as np
sys.path.append('../')
import constants
from utils import (
	convert_pixel_distance_to_meters,
	convert_meter_to_pixel_distance
)

class MiniCourt:
	def __init__(self, frame):
		self.drawing_rectangle_width = 250
		self.drawing_rectangle_height = 450
		self.buffer = 50
		self.padding_court = 20

		self.set_canvas_background_box_position(frame)
		self.set_mini_court_position()
		self.set_court_draw_key_points()
		self.set_court_lines()

	def convert_meter_to_pixels(self, meters):
		return convert_meter_to_pixel_distance(meters,
											   constants.DOUBLE_LINE_WIDTH,
											   self.court_drawing_width
		)

	def set_court_draw_key_points(self):
		draw_key_points = [0]*28

		# point 0
		draw_key_points[0], draw_key_points[1] = int(self.court_start_x), int(self.court_start_y)
		# point 1
		draw_key_points[2], draw_key_points[3] = int(self.court_end_x), int(self.court_start_y)
		# point 2
		draw_key_points[4] = int(self.court_start_x)
		draw_key_points[5] = self.court_start_y + self.convert_meter_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
		# point 3
		draw_key_points[6] = draw_key_points[0] + self.court_drawing_width
		draw_key_points[7] = draw_key_points[5]
		# point 4
		draw_key_points[8] = draw_key_points[0] +  self.convert_meter_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
		draw_key_points[9] = draw_key_points[1] 
		# #point 5
		draw_key_points[10] = draw_key_points[4] + self.convert_meter_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
		draw_key_points[11] = draw_key_points[5] 
		# #point 6
		draw_key_points[12] = draw_key_points[2] - self.convert_meter_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
		draw_key_points[13] = draw_key_points[3] 
		# #point 7
		draw_key_points[14] = draw_key_points[6] - self.convert_meter_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
		draw_key_points[15] = draw_key_points[7] 
		# #point 8
		draw_key_points[16] = draw_key_points[8] 
		draw_key_points[17] = draw_key_points[9] + self.convert_meter_to_pixels(constants.NO_MANS_LAND_HEIGHT)
		# # #point 9
		draw_key_points[18] = draw_key_points[16] + self.convert_meter_to_pixels(constants.SINGLE_LINE_WIDTH)
		draw_key_points[19] = draw_key_points[17] 
		# #point 10
		draw_key_points[20] = draw_key_points[10] 
		draw_key_points[21] = draw_key_points[11] - self.convert_meter_to_pixels(constants.NO_MANS_LAND_HEIGHT)
		# # #point 11
		draw_key_points[22] = draw_key_points[20] +  self.convert_meter_to_pixels(constants.SINGLE_LINE_WIDTH)
		draw_key_points[23] = draw_key_points[21] 
		# # #point 12
		draw_key_points[24] = int((draw_key_points[16] + draw_key_points[18])/2)
		draw_key_points[25] = draw_key_points[17] 
		# # #point 13
		draw_key_points[26] = int((draw_key_points[20] + draw_key_points[22])/2)
		draw_key_points[27] = draw_key_points[21] 

		self.draw_key_points=draw_key_points

	def set_court_lines(self):
		self.lines = [
			(0, 2),
		    (4, 5),
		    (6,7),
		    (1,3),
		    
		    (0,1),
		    (8,9),
		    (10,11),
		    (10,11),
		    (2,3)
		]

	def set_mini_court_position(self):
		self.court_start_x = self.start_x + self.padding_court
		self.court_start_y = self.start_y + self.padding_court
		self.court_end_x = self.end_x - self.padding_court
		self.court_end_y = self.end_y - self.padding_court
		self.court_drawing_width = self.court_end_x - self.court_start_x

	def set_canvas_background_box_position(self, frame):
		frame = frame.copy()

		self.end_x = frame.shape[1] - self.buffer
		self.end_y = self.buffer + self.drawing_rectangle_height
		self.start_x = self.end_x - self.drawing_rectangle_width
		self.start_y = self.end_y - self.drawing_rectangle_height

	# transparent white background for court
	def draw_background_rectangle(self, frame):
		shapes = np.zeros_like(frame, np.uint8)
		# Draw the rectangle
		cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
		out = frame.copy()
		alpha = 0.5 #  50% transparent
		mask = shapes.astype(bool)
		out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

		return out

	def draw_mini_court(self, frames):
		output_frames = []
		for frame in frames:
			frame = self.draw_background_rectangle(frame)
			output_frames.append(frame)

		return output_frames