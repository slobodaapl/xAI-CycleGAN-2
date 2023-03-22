from typing import Tuple
import javabridge
import bioformats
import atexit
from tqdm import tqdm
from cv2 import resize


def exit_handler():
    javabridge.kill_vm()


atexit.register(exit_handler)

# check if javabridge is already running
if not javabridge.jutil._javabridge.get_vm().is_active():
    javabridge.start_vm(class_path=bioformats.JARS)


def _init_logger():
    """This is so that Javabridge doesn't spill out a lot of DEBUG messages
    during runtime.
    From CellProfiler/python-bioformats.
    """
    root_logger_name = javabridge.get_static_field("org/slf4j/Logger",
                                                   "ROOT_LOGGER_NAME",
                                                   "Ljava/lang/String;")

    root_logger = javabridge.static_call("org/slf4j/LoggerFactory",
                                         "getLogger",
                                         "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                         root_logger_name)

    log_level = javabridge.get_static_field("ch/qos/logback/classic/Level",
                                            "WARN",
                                            "Lch/qos/logback/classic/Level;")

    javabridge.call(root_logger,
                    "setLevel",
                    "(Lch/qos/logback/classic/Level;)V",
                    log_level)


class VSIFile:

    def __init__(self, vsi_file: str,
                 roi_size: Tuple[int, int] = (1024, 1024),
                 target_size: Tuple[int, int] = (256, 256),
                 use_pbar: bool = True):
        """
        This class serves to open and close vsi files, and to extract rois from them, but also to
        serve as a generator for continuous extraction of rois from the slide, and keeping track of
        progress and which slides to skip based on whether they're empty or not.

        :param vsi_file: Absolute path of the vsi file to open
        :param roi_size: Size of the roi to extract from the slide, (height, width)
        :param target_size: What dimensions to rescale extracted roi to, before yielding, (height, weidth).
         If None, no rescaling is done.
        :param use_pbar: Whether to use a progress bar or not
        """

        self.file_path = vsi_file
        self.idx = 0
        self.roi_size = roi_size
        self.target_size = target_size
        self.slide = None
        self.shape = None
        self.max_x_idx = None
        self.max_y_idx = None
        self.num_rois = None

        if use_pbar:
            self.pbar = tqdm()
        else:
            self.pbar = None

    def __enter__(self):
        _init_logger()
        self.slide = bioformats.ImageReader(self.file_path)
        self.shape = self.slide.rdr.getSizeY(), self.slide.rdr.getSizeX(), 3
        self.max_x_idx = self.shape[1] // self.roi_size[1]
        self.max_y_idx = self.shape[0] // self.roi_size[0]
        self.num_rois = self.max_x_idx * self.max_y_idx

        if self.pbar is not None:
            self.pbar.total = self.num_rois
            self.pbar.refresh()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.slide.close()

        if self.pbar is not None:
            self.pbar.close()

    def __del__(self):
        self.slide.close()

        if self.pbar is not None:
            self.pbar.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == self.max_x_idx * self.max_y_idx:
            if self.pbar is not None:
                self.pbar.close()
            raise StopIteration

        y = (self.idx // self.max_x_idx) * self.roi_size[0]
        x = (self.idx % self.max_x_idx) * self.roi_size[1]

        roi = self.get_roi(x, y, self.roi_size[0], self.roi_size[1])
        roi = resize(roi, self.target_size) if self.target_size else roi

        self.idx += 1

        if self.pbar is not None:
            self.pbar.update(1)

        return roi

    def _open_slide(self, perform_init: bool = True):
        self.slide = bioformats.ImageReader(self.file_path, perform_init=perform_init)

    def _close_slide(self):
        self.slide.close()

    def get_roi(self, x: int, y: int, height: int, width: int):
        return self.slide.read(c=0, z=0, t=0, rescale=False, XYWH=(x, y, width, height))

    def get_size(self):
        return self.slide.rdr.getSizeX(), self.slide.rdr.getSizeY()
