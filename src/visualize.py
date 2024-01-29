import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.io as io

from data_explorer import DataExplorer


class FramesUtils:
    def __init__(self, img_path, mask_path, total_frames) -> None:
        self.frame = 0
        self.total_frames = total_frames
        self.cell_frame = 0
        self.cell_label = 0
        self.img_path = img_path
        self.mask_path = mask_path
        self.images = os.listdir(img_path)
        self.masks = os.listdir(mask_path)

    def SetFrame(self, number: int):
        self.frame = number

    def IncreaseFrame(self):
        if self.frame < (self.total_frames - 1):
            self.frame += 1

    def DecreaseFrame(self):
        if self.frame > 0:
            self.frame -= 1

    def IncreaseCellFrame(self):
        if self.cell_frame < (self.total_frames - 1):
            self.cell_frame += 1

    def DecreaseCellFrame(self):
        if self.cell_frame > 0:
            self.cell_frame -= 1

    def GetImageAtCurrentFrame(self):
        return self.images[self.frame]

    def GetMaskAtCurrentFrame(self):
        return self.masks[self.frame]


def print_help():
    text = """
    OH MY GOD
    HELP ME

    keys:
    p ... previous data set frame
    n ... next data set frame
    h ... help
    """
    print(text)


def switch_frame_window():
    image = io.imread(
        f'{utilizer.img_path}/{utilizer.GetImageAtCurrentFrame()}'
    )
    mask = io.imread(
        f'{utilizer.mask_path}/{utilizer.GetMaskAtCurrentFrame()}'
    )

    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(mask)
    fig.suptitle(f"Frame: {utilizer.frame + 1}/{utilizer.total_frames}")
    plt.show()


def switch_cell_window():
    image_cell, mask_cell = explorer.GetCellAtFrame(utilizer.cell_frame,
                                                    utilizer.cell_label)
    ax_cell[0].imshow(image_cell, cmap='gray')
    ax_cell[1].imshow(mask_cell)
    fig_cell.suptitle(f"Frame: {utilizer.frame + 1}/{utilizer.total_frames}")
    plt.show()


def onkey(event):

    if event.key == "n" and utilizer.frame < (utilizer.total_frames - 1):
        utilizer.IncreaseFrame()
        utilizer.IncreaseCellFrame()
        if (event.canvas == fig.canvas):
            print("fig canvas n")
            switch_frame_window()
        if (event.canvas == fig_cell.canvas):
            print("fig cell canvas n")
            switch_cell_window()

    if event.key == "p" and utilizer.frame > 0:
        utilizer.DecreaseFrame()
        utilizer.DecreaseCellFrame()
        if (event.canvas == fig.canvas):
            print("fig canvas p")
            switch_frame_window()
        if (event.canvas == fig_cell.canvas):
            print("fig cell canvas p")
            switch_cell_window()

    if event.key == "h":
        print_help()

    if event.key == "p":
        pass


def onclick(event):
    if event.inaxes == ax[1]:
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            y = int(event.ydata)

            if mask[y, x] != 0:
                # can i do this?
                utilizer.cell_label = mask[y, x]
                utilizer.cell_frame = utilizer.frame

                switch_cell_window()
                print("(x={}, y={}): {} {}".format(x, y,
                                                   "klik na bunku :)",
                                                   utilizer.cell_label))


print("Press <h> for help.")

img_path = "../tests/testdata/images"
mask_path = "../tests/testdata/masks"

explorer = DataExplorer("output")
utilizer = FramesUtils(img_path, mask_path, explorer.GetNumberOfFrames())

image = io.imread(f'{utilizer.img_path}/{utilizer.GetImageAtCurrentFrame()}')
mask = io.imread(f'{utilizer.mask_path}/{utilizer.GetMaskAtCurrentFrame()}')

mpl.rcParams['toolbar'] = 'None'  # turn off matplot GUI toolbar

fig, ax = plt.subplots(1, 2)
fig.suptitle(f"Frame: {utilizer.frame + 1}/{utilizer.total_frames}")

fig.canvas.mpl_connect('key_press_event', onkey)
fig.canvas.mpl_connect('button_press_event', onclick)

ax[0].imshow(image, cmap='gray')
ax[1].imshow(mask)

# would be nice to not show it
fig_cell, ax_cell = plt.subplots(1, 2)
# fig_cell.canvas.mpl_connect('button_press_event', onclick)
fig_cell.canvas.mpl_connect('key_press_event', onkey)

plt.show()
