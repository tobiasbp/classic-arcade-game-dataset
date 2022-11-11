"""
Run through all dataset variations and create a bar graph to compare results.
"""

from math import ceil

import tensorflow as tf
import tensorflow_datasets as tfds

# Helper libraries
#import numpy as np
import matplotlib.pyplot as plt

from dataset import classic_arcade_games

# No of epochs used in training
epochs = 10

# The datasets to use
datasets = [
    "classic_arcade_games/8x8",
    "classic_arcade_games/16x16",
    "classic_arcade_games/32x32",
    "classic_arcade_games/64x64",
]

labels = [
  "amidar",
  "depthcho",
  "digdug",
  "dkong",
  "frogger",
  "galagao",
  "invadrmr",
  "missile1",
  "pacman",
  "qix",
  "rallyx",
]

print(tf.__version__)

# Add result dicts in this array
results = []

# Loop through the datasets  
for dataset_no in range(len(datasets)):
    r = {"dataset": datasets[dataset_no]}

    # Load the data. Split into training and test data. (PrefetchDataset)
    #(ds_train, ds_test) , ds_info = tfds.load(
    (train_images, train_labels), (test_images, test_labels) = tfds.load(
        datasets[dataset_no],
        split=['train[:50%]', 'train[50%:]'],
        #with_info=True,
        # Get a tuple (features, label)
        as_supervised=True,
        # Load all of the data
        batch_size=-1
        )

    # Normalize data to values between 0.0 and 1.0
    train_images = tf.cast(train_images, tf.float32) / 255.0
    test_images = tf.cast(test_images, tf.float32) / 255.0

    # Get the dimensions of the dataset images to feed the model
    image_dimensions = train_images.shape[1:3]

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=image_dimensions),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(labels))
    ])

    # Compile the model
    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # Fit the model to the training data
    model.fit(train_images, train_labels, epochs=epochs)

    # Evaluate the model using the unseen testing data
    r["test_loss"], r["test_acc"] = model.evaluate(test_images,  test_labels, verbose=2)

    results.append(r)


# Create a bar graph of the results
bar_width = 0.4

gridnumber = range(1,len(results)+1)

d_acc = [ r["test_acc"] for r in results ]
b_acc = plt.bar(gridnumber, d_acc, width=bar_width,
    label="Accuracy", align="center")

d_loss = [ r["test_loss"] for r in results ]
b_loss = plt.bar([ p + bar_width for p in gridnumber ], d_loss, color="red", width=bar_width,
    label="Loss", align="center")

# Max value rounde up with 1 decimal
max_value = max(
    ceil(max(d_loss) * 10) / 10,
    ceil(max(d_acc) * 10) / 10
    )

plt.title(f'No of epochs: {epochs}')
plt.ylim([0, max_value])
plt.xlim([0,len(results)+1])
plt.xticks(gridnumber,( r["dataset"].split("/")[-1] for r in results ))
plt.legend()
plt.show()
