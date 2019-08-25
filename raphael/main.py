import numpy as np
import cv2 as cv
import util.imcommons as imcommons

img = cv.imread('resources/images/foto1.jpg')

# exibe a imagem na tela.
adjust = True   # > ajusta de acordo com a resolução do monitor?

imcommons. imshow (img, adjust)
#imcommons. imshow (cv. Canny (img, 75, 75))