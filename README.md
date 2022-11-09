# classic-arcade-game-dataset
A TensorFlow dataset for identifying classic arcade games from screendumps.
All screendumps for a game, is, typically, from the games attract mode.
The sequence could also be from the game being played, of the game only has a single stage/level.
The games are identified using the name of the game's MAME ROM set.

The datasets are built with the [TFDS CLI](https://www.tensorflow.org/datasets/cli).

https://www.tensorflow.org/datasets/add_dataset

last part of sequence is used for testing, and first par for training.
Not good. Should be mixed.

# Supported games

* depthcho: Depth Charge (older)  

# Data

# Raw data
The raw data consists of screendumps captured by MAME and sets of square, grayscale images
of various sizes created with the Python script _scale_screndumps.py_. The images
are stored in these directories:

* _data/mame/original_
* _data/mame/16x16_
* _data/mame/32x32_
* _data/mame/64x64_

## Tensor Flow dataset
Unmodified screendumps from MAME are avilable in _/data/mame/original/<mame_id>/*.png_
