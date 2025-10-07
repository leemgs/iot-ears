import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

# TF1.x + Keras 2.2.4
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from keras.models import load_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', default='Output')
    args = ap.parse_args()

    def load(name):
        return np.loadtxt(os.path.join(args.in_dir, name), delimiter=',')
    x_test = load('x_test.csv').reshape(-1, 30, 32, 1)
    y_test = load('y_test.csv')

    model = load_model(os.path.join(args.in_dir, 'model.h5'), compile=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    # Predictions & confusion matrix
    y_pred = model.predict(x_test, verbose=0)
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true_cls = np.argmax(y_test, axis=1)

    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0,1,2])
    accuracy = accuracy_score(y_true_cls, y_pred_cls)

    # Plot CM
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    cm_path = os.path.join(args.in_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight', dpi=150)

    metrics = {'test_loss': float(loss), 'test_accuracy': float(acc)}
    with open(os.path.join(args.in_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Saved:", cm_path)
    print("Metrics:", metrics)

if __name__ == '__main__':
    main()
