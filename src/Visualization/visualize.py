import matplotlib.pyplot as plt
import skimage.io as io

image = io.imread("./tests/testdata/cell_img.tif")
mask = io.imread("./tests/testdata/cell_mask.tif")

# cell_data_path = "../../output.pkl"

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image, cmap='gray')
ax[1].imshow(mask)


def onclick(event):
    if event.inaxes == ax[1]:
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            y = int(event.ydata)

            if mask[y, x] != 0:
                print("(x={}, y={}): {}".format(x, y, "klik na bunku :)"))


cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
