import numpy as np
import cv2 as cv
import util.imcommons as imcommons

img = cv.imread('../resources/images/foto13.jpg')
img_ = cv. blur (img, (5,5), 0)

# exibe 3D colored scatter plot.
### imcommons. colored_scatter (img_)

# máscara de segmentação por cor: amostra -> (60 74 134)
mask_orange = cv. inRange (img_, (40, 50, 90), (100, 110, 170))
mask = mask_orange #+ mask_another_color + ...

#imcommons. imshow (cv. bitwise_and (img_, img_, mask = mask), True)

squares = imcommons. find_squares (img_)

#imcommons. imshow (imcommons. four_point_transform (img_, squares [0]))
cv. imwrite ('../resultados/foto13_somente_quadrado.png', imcommons. four_point_transform (img_, squares [0]))

sqr_size_normalized = imcommons. imsize (imcommons. four_point_transform (img_, squares [0]))
sqr_size_normalized = [sqr_size_normalized [0] / min (sqr_size_normalized), sqr_size_normalized [1] / min (sqr_size_normalized)]
sqr_size_normalized. sort ()
print (sqr_size_normalized)

cv. drawContours (img_, squares, -1, (0, 255, 0), 3)

#imcommons. imshow (img_, True)
cv. imwrite ('../resultados/foto13_mais_quadrado.png', img_)

### imcommons. imshow (cv. bitwise_and (img_, img_, mask = mask), True)
### cv. imwrite ('../resultados/foto1_segmentada.png', cv. bitwise_and (img_, img_, mask = mask))