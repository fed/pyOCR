# -*- coding: utf-8 *-*
import scipy as sp


def encode(char):
    decimal = ord(unicode(char))
    res = []
    while decimal > 0:
        res.append(decimal % 2)
        decimal = decimal / 2
    while len(res) < 8:
        res.append(0)
    return sp.matrix(sp.array(res)).T


def decode(array):
    r = 0
    base = 1
    for x in array.round():
        r += x * base
        base *= 2
    return chr(r)
