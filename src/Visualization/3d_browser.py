
from mayavi import mlab
from skimage import io

# Load TIFF file
image = io.imread('./tests/testdata_3d/masks/man_seg000.tif')

# Create Mayavi figure
fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))

# Create volume visualization
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(image))

# Adjust volume properties if needed
vol._volume_property.shade = False  # Turn off shading for better visibility

vol._ctf.range = [0, image.max() * 2]  # Doesn't work

# Display result
mlab.show()
