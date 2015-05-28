# -*- coding: utf-8 *-*
from PIL import Image, ImageDraw, ImageFont
import string


def txt2img(label, output_file, fontname="arial.ttf", imgformat="PNG",
            fgcolor=(0, 0, 0), bgcolor=(255, 255, 255),
            rotate_angle=0, font_size=16):
    """Render label as image."""
    # fontname must be abs. path
    font = ImageFont.truetype(fontname, font_size)
    imgOut = Image.new("RGBA", (20, 49), bgcolor)

    lines = label.split('\n')
    w = 0
    h = 0
    y_text = 0

    # calculate space needed to render text
    draw = ImageDraw.Draw(imgOut)
    for line in lines:
        draw = ImageDraw.Draw(imgOut)
        sizex, sizey = draw.textsize(line, font=font)
        if sizex > w:
            w = sizex
        if sizey > y_text:
            y_text = sizey
        h += sizey

    #resize the image
    imgOut = imgOut.resize((w, h))
    draw = ImageDraw.Draw(imgOut)

    #write text
    h = 0
    for line in lines:
        draw.text((0, h), line, fill=fgcolor, font=font)
        h += y_text

    if rotate_angle:
        imgOut = imgOut.rotate(rotate_angle)

    imgOut.save(output_file)


def make_dataset(label, filename, fuente, font_size=16, rotate_angle=0):
    txt2img(label, filename + '.png', fontname=fuente, font_size=font_size,
            rotate_angle=rotate_angle)
    open(filename + '.txt', 'w').write(label)

if __name__ == '__main__':
    chars = ' '.join(list(string.lowercase))
    make_dataset(chars, './samples/lower',
        '/usr/share/fonts/truetype/msttcorefonts/arial.ttf', 24)

    chars = ' '.join(list(string.digits))
    make_dataset(chars, './samples/digits',
        '/usr/share/fonts/truetype/msttcorefonts/arial.ttf', 24)

    chars = 'hola mundo'
    make_dataset(chars, './samples/hello_world',
        '/usr/share/fonts/truetype/msttcorefonts/arial.ttf', 24)