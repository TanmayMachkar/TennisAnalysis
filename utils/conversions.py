def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
	# let us consider
	# 20 (pixels) --> 10.97 (meters)
	# pixel_distance --> ? (x)
	# x = (pixel_distance * 10.97) / 20
	return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels

def convert_meter_to_pixel_distance(meters, reference_height_in_meters, reference_height_in_pixels):
	return (meters * reference_height_in_pixels) / reference_height_in_meters