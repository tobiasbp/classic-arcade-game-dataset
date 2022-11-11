# classic-arcade-game-dataset
A TensorFlow dataset for identifying classic arcade games from sequenses of screendumps.
All screendumps for a game are, typically, from the games attract mode.
The sequence could also be from the game being played, if the game only has a single stage/level.
The games are identified using the name of the game's MAME ROM set.

The datasets are built with the [TFDS CLI](https://www.tensorflow.org/datasets/cli).

https://www.tensorflow.org/datasets/add_dataset

last part of sequence is used for testing, and first par for training.
Not good. Should be mixed.

# Supported games

The dataset contains data for the following games (As named in MAME):
* amidar
* depthcho
* digdug
* dkong
* frogger
* galagao
* invadrmr
* missile1
* pacman
* qix
* rallyx

# Data

## Raw data
Unmodified screendumps from MAME are avilable in _/data/mame/original/<mame_id>/*.png_

## Modified data
Squared, grayscale version in different resolutions, are avaiable in _.zip_ archives.
These images have been created using the script _scale_screndumps.py_.
are stored in these directories:

* _data/mame/8x8.zip_
* _data/mame/16x16.zip_
* _data/mame/32x32.zip_
* _data/mame/64x64.zip_


![Grayscale: 8x8](./figures/8x8_5x5.png)
![Grayscale: 16x16](./figures/16x16_5x5.png)
![Grayscale: 32x32](./figures/32x32_5x5.png)
![Grayscale: 64x64](./figures/64x64_5x5.png)