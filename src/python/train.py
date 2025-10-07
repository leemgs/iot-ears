import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Use TF 1.x APIs explicitly (as in the document)
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import keras
from keras import layers, models, optimizers

def load_csvs(in_dir):
    def load(name):
        return np.loadtxt(os.path.join(in_dir, name), delimiter=',')
    x_train = load('x_train.csv')
    y_train = load('y_train.csv')
    x_val   = load('x_val.csv')
    y_val   = load('y_val.csv')
    return x_train, y_train, x_val, y_val

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(30, 32, 1), data_format='channels_last'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(9, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='JSON config path (overrides training defaults)')
    ap.add_argument('--in_dir', default='Output')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=500)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--momentum', type=float, default=0.9)
    args = ap.parse_args()

    import json
    cfg = None
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as cf:
            cfg = json.load(cf)
    # Override with config if provided
    if cfg:
        args.epochs = cfg.get('training',{}).get('epochs', args.epochs)
        args.batch_size = cfg.get('training',{}).get('batch_size', args.batch_size)
        args.lr = cfg.get('training',{}).get('lr', args.lr)
        args.momentum = cfg.get('training',{}).get('momentum', args.momentum)


    x_train, y_train, x_val, y_val = load_csvs(args.in_dir)
    # reshape to (N, 30, 32, 1)
    x_train_r = x_train.reshape(x_train.shape[0], 30, 32, 1)
    x_val_r   = x_val.reshape(x_val.shape[0], 30, 32, 1)

    model = build_model()
    sgd = optimizers.SGD(lr=args.lr, momentum=args.momentum, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    hist = model.fit(x_train_r, y_train, validation_data=(x_val_r, y_val),
                     batch_size=args.batch_size, epochs=args.epochs, verbose=2)

    # Save model
    out_h5 = os.path.join(args.in_dir, 'model.h5')
    model.save(out_h5)
    print("Saved:", out_h5)

    # Plot loss
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend()
    fig_path = os.path.join(args.in_dir, 'train_val_loss.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    print("Saved:", fig_path)

if __name__ == '__main__':
    main()
