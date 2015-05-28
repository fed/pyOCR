# -*- coding: utf-8 *-*
import image_processing as ip
from mlp import MLP
from utils import decode


def convert(image_file, text_file=None):
    img = ip.get_image(image_file)
    lines = []
    for line in ip.get_lines(img):
        words = []
        for word in ip.get_words(img, line):
            chars = []
            for char in ip.get_chars(img, word):
                c = convert_char(img, char)
                chars.append(c)
            words.append(''.join(chars))
        lines.append(' '.join(words))

    if text_file:
        f = open(text_file, 'w')
        f.write('\n'.join(lines))
        f.close()
    else:
        print '\n'.join(lines)


def convert_char(img, char):
    c = ip.process_char(img, char)
    return decode(network.activate(c))


network = MLP.load('lower2.dmp')

if __name__ == '__main__':
    convert('./samples/otra_prueba.png')
