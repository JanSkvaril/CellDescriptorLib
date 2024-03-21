import napari
from skimage import io
import os

# Directory containing TIFF files
directory = './tests/testdata_3d/masks'

# Get list of TIFF files in the directory
files = [file for file in os.listdir(directory) if file.endswith('.tif')]

# Load all TIFF files into a list
images = [io.imread(os.path.join(directory, file)) for file in files]
print(images)

# Open Napari viewer with the stack of images
viewer = napari.view_image(images, name='Stack')

# Run the viewer
napari.run()
