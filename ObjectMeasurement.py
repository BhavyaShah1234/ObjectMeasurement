import cv2 as cv
import numpy as np


def get_contours(img, canny_threshold=None, show_image=False, min_area=1000, fil=0, draw=False):
	if canny_threshold is None:
		canny_threshold = [50, 50]
	image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	image_blur = cv.GaussianBlur(image_gray, ksize=(5, 5), sigmaX=1)
	image_canny = cv.Canny(image_blur, threshold1=canny_threshold[0], threshold2=canny_threshold[1])
	kernel = np.ones((5, 5))
	image_dilate = cv.dilate(image_canny, kernel=kernel, iterations=3)
	image_eroded = cv.erode(image_dilate, kernel=kernel, iterations=2)
	if show_image:
		cv.imshow('Image Canny', image_eroded)
	contours, _ = cv.findContours(image_eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	final_contours = []
	for i in contours:
		area = cv.contourArea(i)
		if area > min_area:
			perimeter = cv.arcLength(i, closed=True)
			corner_points = cv.approxPolyDP(i, 0.02 * perimeter, closed=True)
			bbox = cv.boundingRect(corner_points)
			if fil > 0:
				if len(corner_points) == fil:
					final_contours.append([len(corner_points), area, corner_points, bbox, i])
			else:
				final_contours.append([len(corner_points), area, corner_points, bbox, i])
	final_contours = sorted(final_contours, key=lambda o: o[1], reverse=True)
	if draw:
		for i in final_contours:
			cv.drawContours(img, i[4], -1, (0, 255, 0), 4)
	return img, final_contours


def reorder(points):
	new_points = np.zeros_like(points)
	points = points.reshape((points.shape[0], points.shape[2]))
	add = points.sum(1)
	new_points[0] = points[np.argmin(add)]
	new_points[3] = points[np.argmax(add)]
	diff = np.diff(points, axis=1)
	new_points[1] = points[np.argmin(diff)]
	new_points[2] = points[np.argmax(diff)]
	return new_points


def warp_image(img, points, paper_w, paper_h, pad):
	points = reorder(points)
	points1 = np.float32(points)
	points2 = np.float32([[0, 0], [paper_w, 0], [0, paper_h], [paper_w, paper_h]])
	matrix = cv.getPerspectiveTransform(points1, points2)
	img_warp = cv.warpPerspective(img, matrix, (paper_w, paper_h))
	img_warp = img_warp[pad:img_warp.shape[0]-pad, pad:img_warp.shape[1]-pad]
	return img_warp


def find_distance(points1, points2):
	return pow((pow((points1[0] - points2[0]), 2) + pow((points1[1] - points2[1]), 2)), 0.5)

# object_height = 12.5
# object_width = 7.5
cap = cv.VideoCapture(0)
webcam = False
paper_width = 210
paper_height = 295
scale = 2

while True:
	if webcam:
		_, image = cap.read()
	else:
		image = cv.imread('Reference1.jpg')
		# image = cv.imread('Reference2.jpg')
		image = cv.resize(image, (0, 0), None, 0.25, 0.25)

	image, cont1 = get_contours(image, canny_threshold=[170, 170], show_image=False, min_area=50000, fil=4, draw=False)
	if len(cont1) != 0:
		biggest_contour = cont1[0][2]
		image_warp = warp_image(image, biggest_contour, paper_width * scale, paper_height * scale, pad=20)
		image_cont, cont2 = get_contours(image_warp, canny_threshold=[50, 50], show_image=False, min_area=1000, fil=4, draw=False)
		if len(cont2) != 0:
			for x in cont2:
				# cv.polylines(image_cont, [x[2]], True, (0, 0, 255), 4)
				new_pts = reorder(x[2])
				object_height = round(find_distance(new_pts[0][0] // scale, new_pts[1][0] // scale), 1)
				object_width = round(find_distance(new_pts[0][0] // scale, new_pts[2][0] // scale), 1)
				x1, y1 = new_pts[0][0][0], new_pts[0][0][1]
				x2, y2 = new_pts[1][0][0], new_pts[1][0][1]
				x3, y3 = new_pts[2][0][0], new_pts[2][0][1]
				# angle1 = math.degrees(math.atan((x2-x1)/(y1-y2)))
				# angle2 = math.degrees(math.atan((x3 - x1) / (y3 - y1)))
				cv.arrowedLine(image_cont, (x1, y1), (x2, y2), (255, 0, 255), 2)
				cv.arrowedLine(image_cont, (x1, y1), (x3, y3), (255, 0, 255), 2)
				# cv.circle(image_warp, (x1, y1), 3, (255, 0, 0), -1)
				# cv.circle(image_warp, (x2, y2), 3, (0, 255, 0), -1)
				# cv.circle(image_warp, (x3, y3), 3, (0, 0, 255), -1)
				cv.putText(image_cont, f'{object_height / 10}cm', (x2 // 4, y2+70), cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 1)
				cv.putText(image_cont, f'{object_width / 10}cm', (x3 // 6, y3-30), cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 1)
		cv.imshow('Img Warp', image_cont)
	cv.waitKey(1)
