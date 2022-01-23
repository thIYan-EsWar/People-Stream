# importing in-built library to perform
# mathematical operations
import math


class EuclideanDistance(object):
	"""
	Here, the Euclidean distance(Minkowski distance of power 2)
	is calculated between the center points to find how closely 
	related two objects are thereby differentiating old and new 
	objects.
	"""
	
	def __init__(self):
		# Instantiating necessary variables
		self.center_points = {}
		self.id = 1

	def update(self, objects: [int]) -> []:
		# To update the the center point lists
		obj_ids = []

		for x, y, w, h in objects:
			# iterating through every points in the
			# given list
			cx, cy = x + w // 2, y + h // 2
			same_object = False

			for key, value in self.center_points.items():
				# Calculating the euclidean distance
				dist = math.hypot(cx - value[0], cy - value[1])

				# if the distance is greater than 35 pixels
				# then the object is different. But this value
				# must be adjusted based on the velocity of the
				# objects
				if dist < 15:
					self.center_points[key] = (cx, cy)
					obj_ids.append([x, y, w, h, key])
					same_object = True

					break

			if not same_object:
				self.center_points[self.id] = (cx, cy)
				obj_ids.append([x, y, w, h, self.id])
				self.id += 1

		# updating the center points
		new_center_points = {}

		for obj_id in obj_ids:
			object_id = obj_id[4]
			center = self.center_points[object_id]
			new_center_points[object_id] = center

		self.center_points = new_center_points
		return obj_ids