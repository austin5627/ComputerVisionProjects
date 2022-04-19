import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

BATCH_SIZE = 32
IMAGE_SIZE = (160, 160)
IMAGE_SHAPE = IMAGE_SIZE + (3, )


def load_data(directory):
    train_data = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.1,
        subset="training",
        seed=123,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.1,
        subset="validation",
        seed=123,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    class_names = train_data.class_names

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_data.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_data.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, class_names


def create_model(base_model):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    preprocess_input = tf.keras.applications.densenet.preprocess_input

    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(13)

    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = prediction_layer(x)
    model = tf.keras.Model(inputs, output)
    return model


def train(train_dataset, validation_dataset):

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        weights="imagenet"
    )

    model = create_model(base_model)

    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    epochs = 10
    fine_tune_epochs = 10

    # Train new layers
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
    )

    # Unfreeze 100 layers for fine tuning
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Re-compile model with unfrozen layers
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    # Continue training with unfrozen layers
    model.fit(
        train_dataset,
        epochs=epochs+fine_tune_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
    )

    # Save model as model.h5
    model.save("model.h5")


def test_model(model, validation_dataset, class_names):

    # Pick 9 images from validation set and get their predictions
    validation_images, validation_labels = validation_dataset.as_numpy_iterator().next()
    predictions = model.predict(validation_images)

    # Plot predictions in 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(validation_images[i].astype(np.uint8))
        ax.set_title(f"Real: {class_names[validation_labels[i]]}, Predicted: {class_names[np.argmax(predictions[i])]}")
        ax.axis("off")
    plt.show()


def main():
    data_dir = './flowers/'
    train_dataset, validation_dataset, class_names = load_data(data_dir)
    # if tf.keras.models.load_model("model.h5"):
    #     model = tf.keras.models.load_model("model.h5")
    #     test_model(model, validation_dataset, class_names)
    train(train_dataset, validation_dataset)


if __name__ == "__main__":
    main()
