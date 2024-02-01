import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.io as io

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(-1, parent_dir)

from data_explorer import DataExplorer
# from menu import Menu, MenuItem, ItemProperties

OPEN = 1
CLOSED = 0

"""
TO-DO:
    - when cell window is open, list descriptors in a menu:
        • MaskDecriptors
        • HistogramDescriptors
        • Moments
        • MomentsCentral
        • MomentsHu
        • GlcmFeatures
        • Granulometry
        • PowerSpectrum
        • Autocorrelation
        • LocalBinaryPattern
    - implement visualization of some descriptors
    mentioned above
    - fix counting frames (move frame counter functions to ifs)
"""


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
        self.frame_fig, self.frame_ax = None, None
        self.cell_fig, self.cell_ax = None, None
        self.cell_window_stat = CLOSED

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


def update_frame_window():
    image = io.imread(
        f'{utilizer.img_path}/{utilizer.GetImageAtCurrentFrame()}'
    )
    mask = io.imread(
        f'{utilizer.mask_path}/{utilizer.GetMaskAtCurrentFrame()}'
    )

    utilizer.frame_ax[0].imshow(image, cmap='gray')
    utilizer.frame_ax[1].imshow(mask)
    utilizer.frame_fig.suptitle(
        f"Frame: {utilizer.frame + 1}/{utilizer.total_frames}"
    )
    plt.draw()


def update_cell_window():
    if utilizer.cell_window_stat is CLOSED:
        print("initializing window")
        utilizer.cell_fig, utilizer.cell_ax = plt.subplots(1, 2)
        utilizer.cell_window_stat = OPEN
        utilizer.cell_fig.canvas.mpl_connect('key_press_event', onkey)
        utilizer.cell_fig.canvas.mpl_connect('close_event', closed_window)
        plt.show()

    image_cell, mask_cell = explorer.GetCellAtFrame(utilizer.cell_frame,
                                                    utilizer.cell_label)
    utilizer.cell_ax[0].imshow(image_cell, cmap='gray')
    utilizer.cell_ax[1].imshow(mask_cell)
    utilizer.cell_fig.suptitle(
        f"Frame: {utilizer.cell_frame + 1}/{utilizer.total_frames}"
    )
    plt.draw()


def onkey(event):
    if event.key == "n":
        utilizer.IncreaseCellFrame()
        if event.canvas == utilizer.frame_fig.canvas:
            utilizer.IncreaseFrame()
            update_frame_window()
        elif event.canvas == utilizer.cell_fig.canvas:
            update_cell_window()
        print("fig canvas n", utilizer.frame, utilizer.cell_frame)

    if event.key == "p":
        utilizer.DecreaseCellFrame()
        if event.canvas == utilizer.frame_fig.canvas:
            utilizer.DecreaseFrame()
            update_frame_window()
        elif event.canvas == utilizer.cell_fig.canvas:
            update_cell_window()
        print("fig canvas p", utilizer.frame, utilizer.cell_frame)

    if event.key == "h":
        print_help()


def onclick(event):
    if event.inaxes == utilizer.frame_ax[1]:
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            y = int(event.ydata)

            if mask[y, x] != 0:
                utilizer.cell_label = mask[y, x]
                utilizer.cell_frame = utilizer.frame

                update_cell_window()
                print("(x={}, y={}): {} {}".format(x, y,
                                                   "klik na bunku :)",
                                                   utilizer.cell_label))
                print(explorer.GetDescriptorsForCell(utilizer.cell_frame,
                                                     utilizer.cell_label).keys())


def closed_window(event):
    print("window closed")
    utilizer.cell_window_stat = CLOSED
    return


print("Press <h> for help.")

img_path = "./tests/testdata/images"
mask_path = "./tests/testdata/masks"

explorer = DataExplorer("output")
utilizer = FramesUtils(img_path, mask_path, explorer.GetNumberOfFrames())

image = io.imread(f'{utilizer.img_path}/{utilizer.GetImageAtCurrentFrame()}')
mask = io.imread(f'{utilizer.mask_path}/{utilizer.GetMaskAtCurrentFrame()}')

mpl.rcParams['toolbar'] = 'None'  # turn off matplot GUI toolbar

utilizer.frame_fig, utilizer.frame_ax = plt.subplots(1, 2)
utilizer.frame_fig.suptitle(
    f"Frame: {utilizer.frame + 1}/{utilizer.total_frames}"
)

utilizer.frame_fig.canvas.mpl_connect('key_press_event', onkey)
utilizer.frame_fig.canvas.mpl_connect('button_press_event', onclick)

utilizer.frame_ax[0].imshow(image, cmap='gray')
utilizer.frame_ax[1].imshow(mask)

plt.show()
