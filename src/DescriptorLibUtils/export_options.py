
class ExportOptions:
    """
    Class for export options.
    """
    def __init__(self):
        self.__export_json = 1
        self.__export_pickle = 1
        self.__export_region_imgs = 0
        self.__mode_3d = 0
        self.__remove_background = 0
        self.__crop_offset = 0

    def __str__(self):
        return (f"Export JSON: {self.export_json}\n"
                f"Export pickle: {self.export_pickle}\n"
                f"Export region images: {self.export_region_imgs}\n"
                f"3D mode: {self.mode_3d}\n"
                f"Remove background: {self.remove_background}\n"
                f"Crop offset: {self.crop_offset}\n")

    # setters
    def set_export_json(self, value: int) -> None:
        self.__export_json = value

    def set_export_pickle(self, value: int) -> None:
        self.__export_pickle = value

    def set_export_region_imgs(self, value: int) -> None:
        self.__export_region_imgs = value

    def set_mode_3d(self, value: int) -> None:
        self.__mode_3d = value

    def set_remove_background(self, value: int) -> None:
        self.__remove_background = value

    def set_crop_offset(self, value: int) -> None:
        self.__crop_offset = value

    # getters
    def get_export_json(self) -> int:
        return self.__export_json

    def get_export_pickle(self) -> int:
        return self.__export_pickle

    def get_export_region_imgs(self) -> int:
        return self.__export_region_imgs

    def get_mode_3d(self) -> int:
        return self.__mode_3d

    def get_remove_background(self) -> int:
        return self.__remove_background

    def get_crop_offset(self) -> int:
        return self.__crop_offset
