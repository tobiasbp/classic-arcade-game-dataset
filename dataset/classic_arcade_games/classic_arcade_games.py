"""classic_arcade_games dataset."""

from dataclasses import dataclass
from typing import Tuple

import tensorflow_datasets as tfds

# TODO(classic_arcade_games): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(classic_arcade_games): BibTeX citation
_CITATION = """
@ONLINE {tbp-cagd,
    author = "Tobias Balle-Petersen",
    title  = "Classic Arcade Game Dataset",
    year   = "2022",
    url    = "https://github.com/tobiasbp/classic-arcade-game-dataset"
}
"""

@dataclass
class MyDatasetConfig(tfds.core.BuilderConfig):
  img_size: Tuple[int, int] = (16, 16)

class ClassicArcadeGames(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for classic_arcade_games_16x16 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  BUILDER_CONFIGS = [
      # `name` (and optionally `description`) are required for each config
      MyDatasetConfig(name='16x16', description='Small ...', img_size=(16, 16)),
      MyDatasetConfig(name='32x32', description='Medium ...', img_size=(32, 32)),
      MyDatasetConfig(name='64x64', description='Large ...', img_size=(64, 64)),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(classic_arcade_games): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # 16 x 16 images with 1 color channel
            # shape=self.builder_config.img_size)
            'image': tfds.features.Image(shape=self.builder_config.img_size + (1,)),
            #'image': tfds.features.Image(shape=(16, 16, 1)),
            # Get the labels from a file
            'label': tfds.features.ClassLabel(names_file='label_names.txt'),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        #supervised_keys=('image', 'label'),  # Set to `None` to disable
        supervised_keys=None,
        homepage='https://github.com/tobiasbp/classic-arcade-game-dataset/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(classic_arcade_games): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://github.com/tobiasbp/classic-arcade-game-dataset/raw/main/data/mame/16x16.zip')
    path = dl_manager.download_and_extract(f'https://github.com/tobiasbp/classic-arcade-game-dataset/raw/main/data/mame/{self.builder_config.name}.zip')

    # TODO(classic_arcade_games): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        #'train': self._generate_examples(path / 'train_imgs'),
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(classic_arcade_games): Yields (key, example) tuples from the dataset
    for idx, f in enumerate(path.glob('*/*.png')):
      yield idx, {
          'image': f,
          'label': f.parent.name,
      }
