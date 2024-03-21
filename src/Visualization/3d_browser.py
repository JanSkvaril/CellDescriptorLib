import napari
from skimage import io
# Directory containing TIFF files
directory = './tests/testdata_3d/masks'

# Load TIFF file
image = io.imread('./tests/testdata_3d/masks/man_seg000.tif')

# Open napari viewer
viewer = napari.view_image(image)

# Run the viewer
napari.run()
