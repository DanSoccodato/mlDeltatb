import numpy
import tensorflow as tf
from tensorflow.keras import layers


class TBNN(tf.keras.Model):
    def __init__(self, output_nodes, layer_nodes):
        super(TBNN, self).__init__()
        self._layers = []
        for n in layer_nodes:
            self._layers.append(layers.Dense(n, activation='relu'))

        self._layers.append(layers.Dense(output_nodes, activation='linear'))

    def call(self, inputs):
        for layer in self._layers:
            inputs = layer(inputs)

        return inputs

    def getModelShapes(self):
        model_shapes = [w.shape for w in self.get_weights()]
        return model_shapes

    def getTotalNumberOfWeights(self):
        total_number_of_weights = sum([numpy.prod(s) for s in self.getModelShapes()])
        return total_number_of_weights


class RegularizedShallowNN(tf.keras.Model):
    def __init__(self):
        super(RegularizedShallowNN, self).__init__()
        self.dense_1 = layers.Dense(8, activation='relu')
        self.dense_2 = layers.Dense(8, activation='relu')
        self.dense_3 = layers.Dense(4, activation='linear')
        self.dropout = layers.Dropout(0.5)

    def call(self, inputs, training=True):
        x = self.dense_1(inputs)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense_2(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense_3(x)
        return x

    def getModelShapes(self):
        model_shapes = [w.shape for w in self.get_weights()]
        return model_shapes

    def getTotalNumberOfWeights(self):
        total_number_of_weights = sum([numpy.prod(s) for s in self.getModelShapes()])
        return total_number_of_weights
