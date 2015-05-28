# -*- coding: utf-8 *-*
from mlp import *
import functions
from image_processing import *
from utils import encode, decode


def make_set(image_file, text_file):
    result = []
    chars = {}

    #make a list of chars
    targets = open(text_file, 'r').read()
    targets = targets.replace('\n', '').replace(' ', '')
    targets = list(targets)

    #find the image of every char
    img = get_image(image_file)
    for line in get_lines(img):
        for word in get_words(img, line):
            for char in get_chars(img, word):
                _input = process_char(img, char)
                _target = targets.pop(0)
                chars[_target] = None
                _target = encode(_target)
                result.append((_input, _target))
    #All targets was found
    assert len(targets) == 0
    return result, len(chars)


def train(image_file, text_file, dump_file=None, factor=0.25):
    training_set, char_count = make_set(image_file, text_file)

    number_of_inputs = len(training_set[0][0])
    number_of_outputs = len(training_set[0][1])

    fs, ds = functions.functions['linear']
    fi, di = functions.functions['tanh']

    l1 = Layer(number_of_inputs, char_count * factor, fi, di)
    l2 = Layer(char_count * factor, number_of_outputs, fs, ds)
    network = MLP(l1, l2)

    trainer = BackPropagationTrainer(training_set, adapt_rate=True,
                                    learn_rate=0.8)
    trainer.loop_train(network, max_iter=100000, max_error=0.0001)
    if dump_file:
        network.dump(dump_file)
    trainer.show_error_evolution()


def test(image_file, text_file, dump_file):
    network = MLP.load(dump_file)
    training_set, char_count = make_set(image_file, text_file)

    hits = fails = 0

    for x, t in training_set:
        y = decode(network.activate(x))
        t = decode(t)
        if (y != t):
            fails += 1
        else:
            hits += 1
        print 'Expected: %s - Obtained: %s' % (t, y)
    print 'Hit percent: %5.2f %%' % (float(hits) / (hits + fails) * 100)


if __name__ == '__main__':
    #train('./samples/digits.png', './samples/digits.txt', 'digits.dmp', 1)
    #test('./samples/digits.png', './samples/digits.txt', 'digits.dmp')
    train('./samples/lower.png', './samples/lower.txt', 'lower2.dmp', 0.25)
    test('./samples/lower.png', './samples/lower.txt', 'lower2.dmp')
