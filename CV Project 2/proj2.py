import tensorflow as tf
import argparse

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


def train(train_dataset, validation_dataset, model_file):

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

    # Save model
    model.save(model_file)


def main():
    # Data directory, model
    parser = argparse.ArgumentParser(description="Transfer Learning Test")
    parser.add_argument("--data_dir", type=str, default="./flowers/", help="Data directory")
    parser.add_argument("--model_file", type=str, default="model.h5", help="Name of saved model file")

    args = parser.parse_args()
    data_dir = args.data_dir
    model_file = args.model_file

    train_dataset, validation_dataset, class_names = load_data(data_dir)
    train(train_dataset, validation_dataset, model_file)


if __name__ == "__main__":
    main()
