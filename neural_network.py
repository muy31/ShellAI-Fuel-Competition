# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/muyi/shell_competition/neural_network.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2025-07-18 11:21:27 UTC (1752837687)
import tensorflow as tf

class NeuralNetwork:

    def __init__(self, shape, output_dim, hidden_layers=[64, 32], activation='relu', learning_rate=0.001, epochs=10, batch_size=32, conv_layers=None, l2_reg=0.001, dropout_rate=0.2):
        """
        Args:
            input_shape (tuple): Shape of the input data, e.g., (5, 11, 1) or (50,).
            output_dim (int): Number of output values.
            hidden_layers (list): List of integers, each representing the number of units in a hidden layer.
            activation (str): Activation function for hidden layers.
            learning_rate (float): Learning rate for optimizer.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            conv_layers (list): List of dicts, each dict contains Conv2D layer parameters (e.g., filters, kernel_size, activation, etc.).
            l2_reg (float): L2 regularization strength for Dense layers.
            dropout_rate (float): Dropout rate after each Dense layer (except output).
        """
        self.input_shape = shape
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.conv_layers = conv_layers or []
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(shape=self.input_shape))
        if self.conv_layers:
            for conv in self.conv_layers:
                model.add(tf.keras.layers.Conv2D(**conv))
            model.add(tf.keras.layers.Flatten())
        else:
            model.add(tf.keras.layers.Flatten()) if len(self.input_shape) > 1 else None
        for i, units in enumerate(self.hidden_layers):
            model.add(tf.keras.layers.Dense(units, activation=self.activation, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)))
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(self.output_dim, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
        return model

    def train(self, train_dataset):
        self.model.fit(train_dataset, epochs=self.epochs, batch_size=self.batch_size)

    def evaluate(self, test_dataset):
        results = self.model.evaluate(test_dataset, batch_size=self.batch_size, return_dict=True)
        print('Evaluation results:', results)
        return results