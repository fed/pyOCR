# -*- coding: utf-8 -*-
import scipy as sp
import numpy
import functions
import pylab
import pickle

mul = sp.multiply


class Layer(object):
    def __init__(self, n_inputs, n_outputs, activation_function,
                    derived_function):
        # W = matriz de pesos, BIAS = otra matriz
        self.BIAS = sp.random.random(size=(n_outputs, 1)) / n_inputs
        # Inicializa una matriz con numeros aleatorios
        # Como parametro le paso el numero de elementos que quiero que tenga
        # Este tamaño depende de la cantidad de entradas
        self.W = sp.random.random(size=(n_outputs, n_inputs)) / n_inputs
        # Funcion de activacion pasa como parametro
        self.activation_function = activation_function
        # Derivada de la funcion de activacion pasa como parametro
        self.derived_function = derived_function
        # Vectorizacion = SciPy permite crear una matriz
        self.vectorize_functions()
        # Reseteo los deltas = matriz que uso para ir corrigiendo el error
        self.reset_deltas()

    def vectorize_functions(self):
        self.af = sp.vectorize(self.activation_function, otypes=[numpy.float])
        self.df = sp.vectorize(self.derived_function, otypes=[numpy.float])

    # Cuando le presentamos la entrada y la neurona calcula las salidas
    def activate(self, X):
        self.last_inputs = X
        # Sumatoria, esto pasa por la funcion de activacion
        # W*X producto de matrices
        s = (self.W * X) + self.BIAS
        self.last_outputs = self.af(s)
        self.last_derivatives = self.df(self.last_outputs)
        return self.last_outputs

    def reset_deltas(self):
        self.delta_w = sp.zeros(self.W.shape)
        self.delta_BIAS = sp.zeros(self.BIAS.shape)


# Clase MLP = Red neuronal completa = lista de Layers (Capas)
class MLP(object):
    def __init__(self, *layers):
        self.layers = list(layers)

    # La funcion de activacion de la red completa va a ser las aplicaciones
    # sucesivas de las funciones de activacion de cada uno de las layers que
    # componen a la red. Activar neurona = le presento
    # las entradas, y la neurona calcula las salidas
    def activate(self, inputs):
        i = inputs
        # i es lo que pasamos de entrada a la red, y dps i son las salidas
        # de la primer capa, que son las entradas de la segunda capa.
        # La salida final tambien es i
        for l in self.layers:
            i = l.activate(i)
        return i

    def reset_deltas(self):
        for l in self.layers:
            l.reset_deltas()

    # Guardo en un archivo la red neuronal que YA aprendio, para que la proxima
    # vez que quiera reconocer caracteres no tenga que volver a entrenar la red
    def dump(self, filename):
        for l in self.layers:
            l.af = None
            l.df = None

        f = open(filename, 'wb')
        try:
            pickle.dump(self, f)
        finally:
            f.close()

    def vectorize_functions(self):
        for layer in self.layers:
            layer.vectorize_functions()

    @classmethod
    # Levantar la red neuronal que ya aprendio, para usarla
    def load(cls, filename):
        f = open(filename, 'r')
        try:
            res = pickle.load(f)
            res.vectorize_functions()
        finally:
            f.close()
        return res


# Algoritmo de Entrenamiento
# Conjunto de patrones de entrada con sus salidas esperadas, iterativamente
# va a ir calculando la salida, calculando el error, modifica los pesos.
# Despues vuelve a entrenar, y asi sucesivamente.
class BackPropagationTrainer(object):
    def __init__(self, estimation_subset, **kw_args):
        # eta = velocidad con la cual yo aprendo = tasa de aprendizaje
        self.learn_rate = kw_args.get('learn_rate', 0.5)
        # Parametro booleano que dice si tengo que adaptar
        # esta velocidad (eta) o no
        self.adapt_rate = kw_args.get('adapt_rate', True)
        # Lista de errores
        self.estimation_errors = []
        self.errors_es = []
        # Conjunto de entrenamiento
        self.estimation_subset = estimation_subset

    # Por cada X (entrada del patron), activo la red neuronal
    # Esta es la parte en que voy para adelante (forward feed)
    def train(self, network, training_set):
        ##reset deltas from previous train
        network.reset_deltas()

        for x, t in training_set:
            #forward feed
            y = network.activate(x)

            # Error = diferencia entre lo que obtuve (Y)
            # y lo que esperaba obtener (TARGET)
            e = t - y

            # Aplico la formula del algoritmo de Back Propagation
            for layer in network.layers[::-1]:
                ro = mul(layer.last_derivatives, e)
                # delta_w y delta_BIAS van a tener la acumulacion de todos
                # los delta por cada patron de entrenamiento que tenga
                # en mi training set
                layer.delta_w += mul(ro, layer.last_inputs.T)
                layer.delta_BIAS += ro
                e = layer.W.T * ro

        # weight updates
        # se actualizan los pesos en funcion de los Delta calculados
        for l in network.layers:
            lts = len(training_set)
            l.W = l.W + self.learn_rate * ((l.delta_w / lts))
            l.BIAS = l.BIAS + self.learn_rate * ((l.delta_BIAS / lts))
            # Estoy dividiendo por el largo del training set = Error Promedio
            # de correr todo el training set. Voy a tratar que, en promedio,
            # en todo el training set el error disminuya

    # Actualizamos el valor de eta = tasa de aprendizaje
    def adapt_learn_rate(self, previous_error, current_error):
        if (previous_error > current_error):
            self.learn_rate *= 1.03
        else:
            self.learn_rate *= 0.5

    # Ejecutamos muchas veces el ciclo de entrenamiento (metodo TRAIN)
    # Hasta que de tantas iteraciones como nosotros especifiquemos (max_iter)
    # o hasta que se alcance un error minimo (max_error) que determinemos
    def loop_train(self, network, **kw_args):
        max_iter = kw_args.get('max_iter', 100)
        max_error = kw_args.get('max_error', 1e-20)
        error = functions.error_avg(network, self.estimation_subset)
        self.estimation_errors.append(error)

        self.iterations = 0
        while (self.iterations < max_iter and error > max_error):
            self.train(network, self.estimation_subset)
            error = functions.error_avg(network, self.estimation_subset)
            self.estimation_errors.append(error)
            if self.adapt_rate:
                self.adapt_learn_rate(*self.estimation_errors[-2:])
            self.iterations += 1

            if self.iterations % 100 == 0:
                print 'Iteration:', self.iterations, \
                'Error estimacion: %16.11f' % error, \
                'Rate: %12.11f' % self.learn_rate

    def dump(self, filename):
        f = open(filename, 'w')
        try:
            pickle.dump(self, f)
        finally:
            f.close()

    @classmethod
    def load(cls, filename):
        f = open(filename, 'r')
        try:
            res = pickle.load(f)
        finally:
            f.close()
        return res

    # Curva de Aprendizaje
    def show_error_evolution(self, filename=None, from_epoch=0):
        l = self.estimation_errors[from_epoch:]
        t = numpy.arange(from_epoch, from_epoch + len(l))
        s = numpy.array(l)
        pylab.plot(t, s, label='Estimacion')

        pylab.legend()
        pylab.xlabel('Epocas')
        pylab.ylabel(u'Error cuadrático medio')
        pylab.title(u'Curva de aprendizaje')
        pylab.grid(True)
        if filename:
            pylab.savefig(filename)
        pylab.show()


if __name__ == '__main__':
    pass
