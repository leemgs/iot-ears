import os
import argparse
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from keras.models import load_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', default='Output')
    args = ap.parse_args()

    model_path = os.path.join(args.in_dir, 'model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    # Load model
    model = load_model(model_path, compile=False)

    # Representative dataset for INT8
    x_train = np.loadtxt(os.path.join(args.in_dir, 'x_train.csv'), delimiter=',')
    x_train_r = x_train.reshape(x_train.shape[0], 30, 32, 1)

    def rep_dataset():
        for i in range(min(200, len(x_train_r))):
            yield [x_train_r[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path = os.path.join(args.in_dir, 'model.tflite')
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print("Saved:", out_path)

if __name__ == '__main__':
    main()
