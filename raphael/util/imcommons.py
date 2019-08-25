import tkinter
import cv2 as cv

def imsize (img) :
    """
    Retorna o tamanho da imagem.
    """
    return (img. shape [1], img. shape [0])

def imshow (img, adapt = True) :
    """
    Exibe a imagem na tela, enquadrando na resolução do monitor.
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
    cv. imshow ('image', img_)
    cv. waitKey (0)
    cv. destroyAllWindows ()
    return