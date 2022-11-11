# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds

# Helper libraries
#import numpy as np
import matplotlib.pyplot as plt

from dataset import classic_arcade_games

epochs = 5

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

# Add results in this array so we can create a plot.
results = []


# importing the required module
import matplotlib.pyplot as plt
  
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

    print("Shape of training images:", train_images.shape[1:3])


    # Normalize data to values between 0.0 and 1.0
    train_images = tf.cast(train_images, tf.float32) / 255.0
    test_images = tf.cast(test_images, tf.float32) / 255.0

    image_dimensions = train_images.shape[1:3]

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=image_dimensions),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(labels))
    ])

    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # Fit the model to the training data
    fit = model.fit(train_images, train_labels, epochs=epochs)
    
    #plotter = tfdocs.plots.HistoryPlotter(metric = 'loss', smoothing_std=10)
    #plotter.plot(fit)
    #plt.ylim([0.5, 0.7])

    r["test_loss"], r["test_acc"] = model.evaluate(test_images,  test_labels, verbose=2)

    print(f'\nTest accuracy ({datasets[dataset_no]}):', r["test_acc"])

    results.append(r)


print(results)

gridnumber = range(1,len(results)+1)

b1 = plt.bar(gridnumber, [ r["test_acc"] * 100 for r in results ], width=0.4,
                label="Accuracy", align="center")

b2 = plt.bar(gridnumber, [ r["test_loss"] * 100 for r in results ], color="red", width=0.4,
                label="Loss", align="center")

plt.title(f'No of epochs: {epochs}')
#plt.set_xlabel('xlabel')
#ax.set_ylabel('ylabel')

plt.ylim([0,100.0])
plt.xlim([0,len(results)+1])
plt.xticks(gridnumber,( r["dataset"].split("/")[-1] for r in results ))
#plt.xticks(gridnumber)
plt.legend()
plt.show()


exit(0)
print(type(ds_train))
print(ds_train[0])
print(ds_train[1])
assert isinstance(ds_train, tf.data.Dataset)
assert isinstance(ds_test, tf.data.Dataset)


# Alias for function to get label name from int
get_label = ds_info.features['label'].int2str

def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image = image/255.0
    return image, label

# Print info on the dataset
#print(dir(ds_info.features["label"]))
#print(ds_info.features["label"].names)

# Convert values integers to floating point values between 0.0 and 1.0
#ds_train, ds_test = ds_train / 255.0, ds_test / 255.0

# The sets are now MapDataset of (image, label) entries.
#ds_train_n = ds_train.map(normalize)
#ds_test_n = ds_train.map(normalize)

#print(ds_train.image)


"""
figure_size = 5
plt.figure(figsize=(8,8))
plt.suptitle(datasets[dataset_no])

i = 1
for image, label in ds_train_n.take(figure_size**2):
    #print(image)
    #print(type(item), item)
    #print(item.keys())
    #print(type(item['image']))
    #print(type(item['label']))
    #print(item['image'])
    #print(item['label'])
    #print(get_label(label))

    ax = plt.subplot(figure_size, figure_size, i)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.axis("off")
    plt.title(get_label(label), y=-0.3)
    plt.imshow(image)
    #plt.colorbar()
    #plt.grid(False)
    i += 1
#plt.show()

print("Image shape:", image.numpy().shape)
"""
#print(label.numpy())

"""

plt.figure(figsize=(10, 10))
i = 0
for image, label in ds_train_n.take(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(get_label(label))
    plt.axis("off")
    i += 1
"""


"""
image, label = next(iter(ds_train))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
"""

#cag = tfds.load("classic_arcade_games", split="train")

#for i in ds_test:
#    print(i)

#print(ds_train.unbatch())

#for i in range(10):
#    print(ds_train.next())

#for i, l in ds_train.unbatch():
#    print(i, l)

"""
(training_images, training_labels), (test_images, test_labels) =  tfds.as_numpy(tfds.load(
    "classic_arcade_games/32x32", split=['train[:80%]', 'train[80%:]'], 
    batch_size=-1, 
    ))
print(type(training_labels))
"""

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', 
                           input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='Adam', loss='binary_crossentropy', 
              metrics=['accuracy'])
"""

model = tf.keras.Sequential([
    # Flatten image to an array
    tf.keras.layers.Flatten(input_shape=(16, 16)),
    # tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # Last layer needs number of nodes matching no of classes/labels
    #tf.keras.layers.Dense(len(ds_info.features["label"].names))
    tf.keras.layers.Dense(10)
])

model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )

# Train model
#model.fit(train_images, train_labels, epochs=10)


model.fit(
  ds_train,
  #validation_data=ds_train_n,
  epochs=3
)
