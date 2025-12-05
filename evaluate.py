import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# In this function i have loaded the test data from cifar-10 dataset
def load_test_data(): 
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.flatten()
    return x_test, y_test

# In this function i have plotted the confusion matrix for better visualization of results
def plot_confusion_matrix(cm, class_names, filename, normalize=False):
    if normalize:
        cm = cm.astype("float32") / (cm.sum(axis=1, keepdims=True) + 1e-12)
        title = "Normalized Confusion Matrix"
    else:
        title = "Confusion Matrix"

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), color="white" if cm[i, j] > thresh else "black", fontsize=8, ha="center", va="center",)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {title} to {filename}")

# In this function i have saved some sample correct & incorrect predictions made by the model
def save_sample_predictions(
    x, y_true, y_pred, correct=True,
    filename="results/sample_predictions.png",
    num_images=15,
):
    if correct:
        indices = np.where(y_true == y_pred)[0]
        title = "Correct Predictions"
    else:
        indices = np.where(y_true != y_pred)[0]
        title = "Incorrect Predictions"

    if len(indices) == 0:
        print(f"No {'correct' if correct else 'incorrect'} predictions to display.")
        return

    indices = np.random.choice(indices, size=min(num_images, len(indices)), replace=False)
    cols = 5
    rows = int(np.ceil(len(indices) / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x[idx])
        true_label = CLASS_NAMES[y_true[idx]]
        pred_label = CLASS_NAMES[y_pred[idx]]
        plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=8)
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved sample {'correct' if correct else 'incorrect'} predictions to {filename}")


# In this main function i have loaded the test data from cifar-10 dataset and then loaded the trained model from train.py to evaluate the test loss & accuracy
def main():
    
    #  i have loaded the test data
    x_test, y_test = load_test_data()

    # then i loaded the  trained model from tain.py
    model_path = "results/cnn_cifar10_model.h5"
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")

    # used this block for evaluating test loss & accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # predict the probabilities and class labels
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, CLASS_NAMES,
        filename="results/confusion_matrix.png",
        normalize=False
    )

    plot_confusion_matrix(
        cm, CLASS_NAMES,
        filename="results/confusion_matrix_normalized.png",
        normalize=True
    )

    save_sample_predictions(
        x_test, y_test, y_pred,
        correct=True,
        filename="results/correct_examples.png"
    )
    
    save_sample_predictions(
        x_test, y_test, y_pred,
        correct=False,
        filename="results/incorrect_examples.png"
    )

    print("Evaluation complete. Metrics and images saved in 'results/'.")


if __name__ == "__main__":
    main()
