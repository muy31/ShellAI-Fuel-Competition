# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/muyi/shell_competition/train_custom_blendnet.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2025-07-20 21:25:02 UTC (1753046702)

import os
import tensorflow as tf
from csv_tensor_loader import csv_to_tf_dataset

class BlendNet(tf.keras.Model):

    def __init__(self, hidden_units=[16, 16, 16], include_fraction_in_subnet_input=False, **kwargs):
        super().__init__(**kwargs)
        self.include_fraction_in_subnet_input = include_fraction_in_subnet_input
        input_dim = 11 if include_fraction_in_subnet_input else 10
        self.subnets = []
        for _ in range(5):
            layers = [tf.keras.layers.Dense(hidden_units[0], activation='relu'), tf.keras.layers.Dense(hidden_units[1], activation='relu'), tf.keras.layers.Dense(10, activation='linear')]
            self.subnets.append(tf.keras.Sequential(layers))

    def call(self, inputs):
        x = tf.squeeze(inputs, axis=-1)
        comps = tf.split(x, num_or_size_splits=5, axis=1)
        outputs = []
        for i, comp in enumerate(comps):
            comp = tf.squeeze(comp, axis=1)
            if self.include_fraction_in_subnet_input:
                subnet_input = comp
            else:
                subnet_input = comp[:, :10]
            fraction = comp[:, 10:11]
            subnet_out = self.subnets[i](subnet_input)
            tf.debugging.assert_shapes([(subnet_out, ('batch', 10)), (fraction, ('batch', 1))], message=f'subnet_out shape: {subnet_out.shape}, fraction shape: {fraction.shape}, subnet_input shape: {subnet_input.shape}')
            gated = subnet_out * fraction
            outputs.append(gated)
        final = tf.add_n(outputs)
        return final
    

if __name__ == '__main__':
    CSV_PATH = 'dataset/train.csv'
    BATCH_SIZE = 64
    EPOCHS = 1000
    CHECKPOINT_EVERY = 5
    CHECKPOINT_DIR = 'model_checkpoints_blendnet'
    LEARNING_RATE = 0.0001
    INPUT_SHAPE = (5, 11, 1)
    OUTPUT_DIM = 10
    train_ds, test_ds = csv_to_tf_dataset(CSV_PATH, batch_size=BATCH_SIZE, shuffle=True, test_size=0.2, random_state=42)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_epoch_{epoch:02d}.h5')
    model = BlendNet(include_fraction_in_subnet_input=False)
    model.build((None, 5, 11, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0), loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, save_freq=CHECKPOINT_EVERY * len(train_ds), verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    class TrainMetricsCallback(tf.keras.callbacks.Callback):

        def __init__(self, train_ds):
            super().__init__()
            self.train_ds = train_ds

        def on_epoch_end(self, epoch, logs=None):
            results = self.model.evaluate(self.train_ds, verbose=0)
            print(f'\n[Epoch {epoch + 1}] Full training set MAE: {results[1]:.4f}, MSE: {results[2]:.4f}')
    
    
    train_metrics_callback = TrainMetricsCallback(train_ds)
    model.fit(train_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint_callback, train_metrics_callback, early_stopping], validation_data=test_ds, verbose=1)
    print('\nEvaluating on test set:')
    model.evaluate(test_ds, batch_size=BATCH_SIZE)