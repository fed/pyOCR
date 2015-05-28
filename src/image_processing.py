# -*- coding: utf-8 *-*
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as nd
import scipy.misc as misc
import numpy as np
import matplotlib.patches as patches
import scipy as sp


# Paso la imagen a una matriz de pixels
# Abre la imagen, la guarda en una matriz de pixeles
# Por defecto se guarda como escala de grises, cada
# elemento de la matriz es un numero que esta entre 0 y 255
# Threshold to BW: si el valor es mayor a 200 lo paso a 255 (True=Black),
# si es menor lo paso a 0 (False=White). Aplico esta operacion a cada uno
# de los elementos de la matriz
def get_image(path):
    #open the image
    img = misc.imread(path, True)
    #threshold to BW
    img = img > 200
    return img


def get_lines(img):
    '''return a list of slices for every line of text found in the image'''
    h, w = img.shape
    _slice = points_to_slice(0, 0, h, w)
    kernel = np.ones((3, w))
    return get_objects(img, _slice, kernel, False)


def get_words(img, line):
    '''return a list of slices for every word found in the line'''
    h = line[0].stop - line[0].start
    k = np.ones((7, round(h * 0.3)))
    return get_objects(img, line, k)


def get_chars(img, word):
    '''return a list of slices for every word found in the line'''
    return get_objects(img, word, np.ones((7, 1)))


def process_char(img, char_slice):
    '''return an slice of img reshaped to (20, 8) '''
    c = img[char_slice]
    c = misc.imresize(c, (20, 8))
    c = (c > 128).astype(int)
    return sp.matrix(character_property_extraction(c)).T


def get_objects(img, _slice, kernel, original_heigth=True):
    '''return a list of slices '''
    or0, oc0, or1, oc1 = slice_to_points(_slice)
    subgrupo = img[_slice]
    #hacemos un opening para eliminar objetos chicos
    subgrupo = nd.binary_opening(subgrupo, structure=kernel)
    subgrupo = -subgrupo
    k2 = np.ones((3, 3))
    #este kernel es para que label tenga en cuenta hasta los pixeles
    #que se tocan diagonalmente
    objects, num_objects = nd.measurements.label(subgrupo, structure=k2)
    slices = []
    for s in nd.find_objects(objects):
        nr0, nc0, nr1, nc1 = slice_to_points(s)
        if original_heigth:
            abs_slice = points_to_slice(or0, oc0 + nc0, or1, oc0 + nc1)
        else:
            abs_slice = points_to_slice(or0 + nr0, oc0 + nc0,
                                        or0 + nr1, oc0 + nc1)
        slices.append(abs_slice)
    slices.sort(lambda x, y: cmp(x[1].start, y[1].start))
    return slices


def slice_to_points(_slice):
    r, c = _slice[:2]
    r0 = r.start
    r1 = r.stop
    c0 = c.start
    c1 = c.stop
    return (r0, c0, r1, c1)


def points_to_slice(r0, c0, r1, c1):
    return (slice(r0, r1, None), slice(c0, c1, None))


def character_property_extraction(char):
    av = []
    x, y = char.shape
    for c in np.hsplit(char, y / 2):
        for r in np.vsplit(c, x / 2):
            av.append(np.average(r))
    return av


def show_sample(path, interval=200):
    img = get_image(path)
    patch_list = []
    for line in get_lines(img):
        patch_list.append((line, 'blue'))
        for word in get_words(img, line):
            patch_list.append((word, 'red'))
            for char in get_chars(img, word):
                patch_list.append((char, 'green'))
    make_animation(img, patch_list, interval)


def make_animation(img, patch_list, interval=200):

    def add_patch(num, plot, patch_list):
        _slice, color = patch_list[num]
        or0, oc0, or1, oc1 = slice_to_points(_slice)
        p = patches.Rectangle((oc0, or0), oc1 - oc0, or1 - or0, fc='none',
                               ec=color)
        plot.add_patch(p)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cm.Greys_r)

    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig, add_patch, len(patch_list),  # lint:ok
        fargs=(ax, patch_list), interval=interval, blit=False)
    plt.show()


def show_images(images):
    f = plt.figure()
    for i, image in enumerate(images):
        ax = f.add_subplot(len(images), 1, 1 + i)
        ax.imshow(image, cmap=cm.Greys_r)
    plt.show()


def show_filter():
    img = get_image('./samples/muestra.png')
    img2 = nd.binary_opening(img, structure=np.ones((3, 3)))
    img3 = -img2
    show_images([img, img2, img3])

if __name__ == '__main__':
    show_filter()
    show_sample('./samples/muestra.png', 300)
    show_sample('./samples/una_prueba.png', 200)
    show_sample('./samples/otra_prueba.png', 200)