import pandas as pd
import argparse
import tensorflow as tf


def load_model_weights(model, weights=None):
    my_model = tf.keras.models.load_model(model)
    return my_model


def get_images_labels(df, classes, img_height, img_width):
    # map class names to integers
    class_to_idx = {k: v for v, k in enumerate(classes)}

    images = []
    labels = []
    for i, row in df.iterrows():
        image = tf.io.read_file(row['image_path'])
        image = decode_img(image, img_height, img_width)
        label = class_to_idx[row['label']]
        images.append(image)
        labels.append(label)

    images = tf.stack(images)
    labels = tf.stack(labels)

    # Convert the lists into a dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(1)

    return dataset

    # test_images = []
    # test_labels = []
    # for index, row in df.iterrows():
    #     test_labels.append(class_to_idx[row['label']])
    #     img = tf.io.read_file(row['image_path'])
    #     img = decode_img(img, img_height, img_width)
    #     test_images.append(img)
    # return test_images, test_labels


def decode_img(img, img_height, img_width):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning Test")
    parser.add_argument('--model', type=str, default='model.h5', help='Saved model')
    parser.add_argument('--weights', type=str, default=None, help='weight file if needed')
    parser.add_argument('--test_csv', type=str, default='flowers_test.csv', help='CSV file with true labels')

    args = parser.parse_args()
    model = args.model
    test_csv = args.test_csv

    test_df = pd.read_csv(test_csv, skipinitialspace=True)
    classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy',
               'carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip']

    # Rewrite the code to match with your setup
    dataset = get_images_labels(test_df, classes, 160, 160)

    my_model = load_model_weights(model)
    loss, acc = my_model.evaluate(dataset, verbose=1)
    print(f'Test model, accuracy: {100 * acc:5.5f}%')


if __name__ == "__main__":
    main()
