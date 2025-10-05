import tensorflow as tf
import os
import argparse
import matplotlib.pyplot as plt

def main(image):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    image = tf.keras.preprocessing.image.load_img(image, target_size=(28, 28), color_mode="grayscale")
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)

    model_path = "model.keras"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),   tf.keras.layers.Dense(128, activation='relu'),   tf.keras.layers.Dropout(0.2),   tf.keras.layers.Dense(10, activation='softmax') ])
        model.compile(optimizer='adam',               loss='sparse_categorical_crossentropy',               metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5) # użyj verbose=0 jeśli jest problem z konsolą
        model.evaluate(x_test, y_test)
        model.save('model.keras')

        history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)
        plt.figure(figsize=(12,5))
        # Loss
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss curve')
        plt.legend()

        # Accuracy
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy curve')
        plt.legend()
        plt.show()

    # przewidywanie liczby na obrazku na podstawie modelu (wczytanego lub wytrenowanego od początku)
    print(model.predict(image).argmax())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a simple neural network on the MNIST dataset.')
    parser.add_argument('image', type=str)

    args = parser.parse_args()
    main(image=args.image)
