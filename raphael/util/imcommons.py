import tkinter
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def imsize (img) :
    """
    *** Retorna o tamanho da imagem.
    """
    return (img. shape [1], img. shape [0])

def imshow (img, adapt = True) :
    """
    *** Exibe a imagem na tela, enquadrando na resolução do monitor.
    """
    # captura tamanho da tela.
    sc = tkinter. Tk ()
    sc_w = sc. winfo_screenwidth  ()
    sc_h = sc. winfo_screenheight ()
    sc_area = sc_w * sc_h

    # captura tamanho da imagem.
    im_size = imsize (img)
    im_area = im_size [0] * im_size [1]

    # ajusta o tamanho da imagem de acordo com a resolução do monitor.
    if (adapt and im_area > sc_area) :
        im_resize_percent = im_area / sc_area
        im_size = ((int) (im_size [0] / im_resize_percent), (int) (im_size [1] / im_resize_percent))

    img_ = cv. resize (img, im_size)

    # exibe a imagem.
    cv. imshow ('imagem', img_)
    cv. waitKey (0)
    cv. destroyAllWindows ()
    return

def colored_scatter (img) :
    # captura tamanho da tela.
    sc = tkinter. Tk ()
    sc_w = sc. winfo_screenwidth  ()
    sc_h = sc. winfo_screenheight ()
    sc_area = sc_w * sc_h

    # captura tamanho da imagem.
    im_size = imsize (img)
    im_area = im_size [0] * im_size [1]

    # ajusta o tamanho da imagem de acordo com a resolução do monitor.
    if (im_area > sc_area) :
        im_resize_percent = im_area / sc_area
        im_size = ((int) (im_size [0] / im_resize_percent), (int) (im_size [1] / im_resize_percent))

    img_ = cv. resize (img, im_size)

    r, g, b = cv. split (img_)
    fig = plt. figure ()
    axis = fig. add_subplot (1, 1, 1, projection = "3d")

    pixel_colors = img_. reshape ((np.shape(img_)[0]*np.shape(img_)[1], 3))
    norm = colors. Normalize (vmin=-1.,vmax=1.)
    norm. autoscale (pixel_colors)
    pixel_colors = norm (pixel_colors). tolist ()

    axis. scatter (g. flatten (), b. flatten (), r. flatten (), facecolors = pixel_colors, marker = ".")
    axis. set_ylabel ("Green")
    axis. set_zlabel ("Blue")
    axis. set_xlabel ("Red")
    plt. show()
    return

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    xrange = range
    squares = []
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            bin, contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 5000 and cv.contourArea(cnt) < 10000 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv.getPerspectiveTransform(rect, dst)
	warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped