import shutil

from pathlib import Path

import cv2

def resize_image(source: Path, destination: Path, size:int, color = None):
    """
    Resize source image to a square of size <size>.
    """

    # Read source image
    image_original = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)

    # Resize image to a square
    image_new = cv2.resize(image_original, (size, size))

    # Convert color
    if not color is None:
        image_new = cv2.cvtColor(image_new, color)


    # Create destination dir if it does not exist
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Write converted image to disk
    cv2.imwrite(str(destination), image_new)

    print("Converting image:", source, "->", destination)

    assert destination.is_file()

def convert_screendumps(source_dir: Path, destination_dir, size: int, color = cv2.COLOR_BGR2GRAY, create_archive: bool = True):

    # Loop through the gamedirs
    for game_dir in source_dir.iterdir():        
        
        game_name = game_dir.name
        print("Scaling images for game:", game_name)

        # Run through game dirs
        if game_dir.is_dir():

            # Run through screendumps
            for screendump in game_dir.iterdir():
                assert screendump.is_file()
                resize_image(screendump, destination_dir / game_name / screendump.name, size, color)

        else:
            print("Warning: Non dir found in datadir", game_dir)

    if create_archive is True:
        # Create the archive
        shutil.make_archive(base_name=destination_dir,format="zip", root_dir=destination_dir, dry_run=False)
        # Delete the image files now in the archive
        assert destination_dir.is_dir()
        shutil.rmtree(destination_dir)

# Create square gray scale images from screendumps
convert_screendumps(Path("data/mame/original"), Path("data/mame/8x8"), 8)
convert_screendumps(Path("data/mame/original"), Path("data/mame/16x16"), 16)
convert_screendumps(Path("data/mame/original"), Path("data/mame/32x32"), 32)
convert_screendumps(Path("data/mame/original"), Path("data/mame/64x64"), 64)