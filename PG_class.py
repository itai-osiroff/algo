import PySpin
from simple_pyspin import Camera, CameraError
import numpy as np


class PG_controller:
    FULL_FRAME_WIDTH = 5320
    FULL_FRAME_HEIGHT = 4600

    def __init__(self, settings):
        settings = settings or {}
        self.settings = settings
        self.cam = None
        self.Nr = None
        self.Nc = None
        self.small_rois = None
        self.big_roi = None
        self.num_rois = None
        self.wind_rad = None
        self.N_window = None
        self.Nr_big = None
        self.Nc_big = None
        self.single_roi_mode = False
        self.single_roi_request = settings.get("PG_single_roi")
        self.user_single_roi = settings.get("PG_roi")
        self.sensor_width = int(settings.get("PG_sensor_width", self.FULL_FRAME_WIDTH))
        self.sensor_height = int(settings.get("PG_sensor_height", self.FULL_FRAME_HEIGHT))
        self.initialize()
        self.start()
        self.stop()
        self.set_rois()
        self.cam.AdcBitDepth = 'Bit12'
        self.cam.PixelFormat = 'Mono8'

        self.num_mean = settings.get("PG_num_mean", 1)
        self.PG_exp_FLR = settings.get("PG_exp_FLR")
        self.PG_exp_LSR = settings.get("PG_exp_LSR")
        self.set_exposure(settings.get("PG_exp") or self.PG_exp_FLR)
        self.set_gain(settings['PG_gain']) if 'PG_gain' in settings else None

        # self.cam.AcquisitionFrameRateEnable = True
        # self.cam.AcquisitionFrameRate = 1.0

        # # GigE-specific, if exposed by simple_pyspin:
        # self.cam.GevSCPSPacketSize = 9000      # requires jumbo frames enabled on NIC
        # self.cam.GevSCPD = 1000                # inter-packet delay; increase if packets drop

        self.start()

    def __del__(self):
        if self.cam is None:
            return
        try:
            self.set_exposure(80e3)
            self.set_gain(20)
        except Exception:
            pass
        try:
            self.stop()
        except Exception:
            pass
        try:
            self.cam.TriggerMode = 'Off'
            self.cam.close()
        except Exception:
            pass

    def start(self):
        self.cam.start()

    def stop(self):
        self.cam.stop()

    def initialize(self):
        self.cam = Camera()
        self.cam.init()
        self.cam.TriggerMode = 'On'
        self.cam.TriggerSource = 'Software'

    def set_exposure(self, new_exp):
        self.cam.ExposureAuto = 'Off'
        self.cam.ExposureTime = new_exp
        # Validate exposure change
        cam_exp = self.cam.ExposureTime
        if cam_exp / new_exp > 1.5 or cam_exp / new_exp < (2 / 3):
            print('Supposed to be ' + str(new_exp) + ', but the exposure is ' + str(cam_exp))

    def set_gain(self, new_gain):
        self.cam.GainAuto = 'Off'
        self.cam.Gain = new_gain

    def set_gamma(self, new_gamma):
        self.cam.GammaEnable = True
        if new_gamma > 4:
            print("gamma too big, setting to 4")
            new_gamma = 4
        if new_gamma < 0.25:
            print("gamma too small, setting to 0.25")
            new_gamma = 0.25
        self.cam.Gamma = new_gamma

    @staticmethod
    def _coerce_rois(rois):
        rois_arr = np.asarray(rois, dtype=int)
        if rois_arr.ndim == 1:
            rois_arr = rois_arr.reshape(1, -1)
        if rois_arr.shape[1] != 4:
            raise ValueError("ROI should have 4 integers: [r1, r2, c1, c2].")
        return rois_arr

    def _roi_to_slices(self, roi, frame_shape=None):
        if frame_shape is None:
            frame_shape = (self.Nr, self.Nc)
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        if frame_h <= 0 or frame_w <= 0:
            raise ValueError("Invalid frame shape for ROI slicing.")

        r1, r2, c1, c2 = [int(v) for v in roi]
        if r2 < r1:
            r1, r2 = r2, r1
        if c2 < c1:
            c1, c2 = c2, c1

        r1 = int(np.clip(r1, 0, frame_h - 1))
        r2 = int(np.clip(r2, 0, frame_h - 1))
        c1 = int(np.clip(c1, 0, frame_w - 1))
        c2 = int(np.clip(c2, 0, frame_w - 1))

        return r1, r2 + 1, c1, c2 + 1

    def _roi_shape(self, roi, frame_shape=None):
        r1, r2, c1, c2 = self._roi_to_slices(roi, frame_shape=frame_shape)
        return r2 - r1, c2 - c1

    def set_rois(self):
        loaded_small_rois, loaded_wind_rad, loaded_big_roi = PG_load_ROI()
        small_rois = self._coerce_rois(loaded_small_rois)

        if self.user_single_roi is not None:
            small_rois = self._coerce_rois(self.user_single_roi)

        if self.single_roi_request is None:
            self.single_roi_mode = len(small_rois) == 1
        else:
            self.single_roi_mode = bool(self.single_roi_request)

        if self.single_roi_mode:
            if self.user_single_roi is not None:
                single_roi = small_rois[0]
            elif loaded_big_roi is not None:
                single_roi = np.asarray(loaded_big_roi, dtype=int).reshape(4)
            else:
                single_roi = small_rois[0]
            self.small_rois = self._coerce_rois(single_roi)
            self.big_roi = np.asarray(self.small_rois[0], dtype=int)
        else:
            self.small_rois = small_rois
            self.big_roi = np.asarray(loaded_big_roi, dtype=int).reshape(4)

        self.num_rois = len(self.small_rois)
        self.wind_rad = int(loaded_wind_rad) if loaded_wind_rad is not None else None

        is_started = self.is_started()
        if is_started:
            self.cam.stop()
        self.cam.Width = self.sensor_width
        self.cam.Height = self.sensor_height
        self.Nc = self.cam.Width
        self.Nr = self.cam.Height
        if is_started:
            self.cam.start()

        self.Nr_big, self.Nc_big = self._roi_shape(self.big_roi, frame_shape=(self.Nr, self.Nc))
        if self.single_roi_mode:
            self.N_window = self.Nr_big
        elif self.wind_rad is not None:
            self.N_window = self.wind_rad * 2 + 1
        else:
            self.N_window = self.Nr_big

    def get_frames(self, N_mean=None, mean_flag=True, split=True, full_screen=False):
        if N_mean is None:
            N_mean = self.num_mean
        frames = np.empty([self.Nr, self.Nc, N_mean], dtype=np.uint16)
        for i in range(N_mean):
            self.cam.TriggerSoftware()
            frame = self.cam.get_array()
            frames[:, :, i] = frame
        if split:
            frames = self.split_to_images(frames)
            mean_axis = frames.ndim - 1
        elif full_screen:
            mean_axis = 2
        else:
            r1, r2, c1, c2 = self._roi_to_slices(self.big_roi, frame_shape=frames.shape[:2])
            frames = frames[r1:r2, c1:c2, :]
            mean_axis = 2
        if mean_flag:
            frames = np.mean(frames, axis=mean_axis)
        return frames

    def get_frame(self, split=False, full_screen=False):
        """Capture and return a single 2D frame using current ROI behavior."""
        self.cam.TriggerSoftware()
        frame = self.cam.get_array()

        if split:
            split_frame = self.split_to_images(frame)
            split_frame = np.asarray(split_frame)
            split_frame = np.squeeze(split_frame)
            if split_frame.ndim != 2:
                raise ValueError(
                    "split=True returned multiple ROI images; cannot represent as one 2D frame."
                )
            return split_frame

        if full_screen:
            return frame

        r1, r2, c1, c2 = self._roi_to_slices(self.big_roi, frame_shape=frame.shape[:2])
        return frame[r1:r2, c1:c2]

    def is_started(self):
        # is_started = 0
        # try:
        #     self.cam.Width = self.cam.Width
        # except CameraError:
        #     is_started = 1
        # return is_started
        return self.cam.running

    def get_exp(self):
        return self.cam.ExposureTime

    def get_gain(self):
        return self.cam.Gain

    def split_to_images(self, I):
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
        roi_slices = [self._roi_to_slices(roi, frame_shape=I.shape[:2]) for roi in self.small_rois]
        roi_h = roi_slices[0][1] - roi_slices[0][0]
        roi_w = roi_slices[0][3] - roi_slices[0][2]

        for r1, r2, c1, c2 in roi_slices[1:]:
            if (r2 - r1) != roi_h or (c2 - c1) != roi_w:
                raise ValueError("All ROIs must have the same shape to stack into one array.")

        images = np.empty([roi_h, roi_w, self.num_rois, I.shape[2]], dtype=np.uint16)
        for i_roi, (r1, r2, c1, c2) in enumerate(roi_slices):
            images[:, :, i_roi, :] = I[r1:r2, c1:c2, :]

        if self.single_roi_mode and self.num_rois == 1:
            return images[:, :, 0, :]

        return images

    def get_unsaturated_photo(self, N_mean=None, split_flag=True, full_screen=False):
        if N_mean is None:
            N_mean = self.num_mean
        counter = 0
        div_num = 1.5
        min_camera_exp = 50
        original_exp = self.get_exp()
        while True:
            I = self.get_frames(N_mean=1, mean_flag=True, split=split_flag, full_screen=full_screen)
            if I.max() > 60e3:
                counter += 1
                new_exp = round(original_exp / (div_num ** counter))
                if new_exp < min_camera_exp:
                    print('Saturated photo')
                    counter -= 1
                    break
                self.set_exposure(new_exp)
            else:
                I = self.get_frames(N_mean=N_mean, mean_flag=True, split=split_flag, full_screen=full_screen)
                break
        # I = self.get_frames(N_mean=N_mean, mean_flag=True)
        I *= (div_num ** counter)
        self.set_exposure(original_exp)
        return I

    def get_HDR_photo(self, N_mean=None, exp_arr=None):
        if N_mean is None:
            N_mean = self.num_mean

        original_exp = self.get_exp()

        if exp_arr is None:
            multipliers = np.array([0.05, 0.3, 0.66, 1, 2, 4], dtype=float)
            exp_arr = original_exp * multipliers
        else:
            exp_arr = np.asarray(exp_arr, dtype=float)

        hdr_accum = None
        weight_accum = None

        try:
            for exp in exp_arr:
                self.set_exposure(exp)
                img_uint16 = self.get_frames(
                    N_mean=N_mean,
                    mean_flag=True,
                    split=False
                )

                max_val = np.iinfo(np.uint16).max  # 65535
                img = img_uint16.astype(np.float32)
                img_norm = img / max_val
                valid = (img_uint16 > 50) & (img_uint16 < max_val)
                w = 1.0 - 2.0 * np.abs(img_norm - 0.5)
                w *= valid

                # Radiance estimate
                img_radiance = img_norm / exp

                if hdr_accum is None:
                    hdr_accum = img_radiance * w
                    weight_accum = w
                else:
                    hdr_accum += img_radiance * w
                    weight_accum += w

            weight_accum = np.clip(weight_accum, 1e-8, None)
            hdr = hdr_accum / weight_accum

        finally:
            self.set_exposure(original_exp)

        return hdr

def PG_load_ROI():
    # r0s, c0s = get_ROI_params()
    # num_rois = len(r0s)
    # small_rois = []
    # rad = int(np.mean(np.diff(c0s)) / 2)
    # for i_roi in range(num_rois):
    #     r1 = r0s[i_roi] - rad
    #     r2 = r0s[i_roi] + rad
    #     c1 = c0s[i_roi] - rad
    #     c2 = c0s[i_roi] + rad
    #     small_rois.append([r1, r2, c1, c2])
    # small_rois_temp = np.array(small_rois)  # convert to NumPy array

    # # New variable: [min(r1), max(r2), min(c1), max(c2)]
    # overall_roi = [small_rois_temp[:, 0].min(),  # min of r1s
    #                small_rois_temp[:, 1].max(),  # max of r2s
    #                small_rois_temp[:, 2].min(),  # min of c1s
    #                small_rois_temp[:, 3].max()]  # max of c2s
    overall_roi = np.array([1050, 1050+3252, 600, 600+3104], dtype=int)
    small_rois = np.array([1050, 1050+3252, 600, 600+3104], dtype=int)
    rad = int(3252 / 2)  # Radius of ROI is half the short side of the camera
    return small_rois, rad, overall_roi


def get_param(text, name):
    return int(text[text.index(name) + 1])


def get_ROI_params():
    path = r"C:\Users\user\OneDrive - Technion\Research project 4 Confocal Microscope\System_Lenslets\_Parameters\Val_00.txt"
    with open(path, "r") as f:
        text = f.read().split()
    r0s_index = text.index("r0s")
    c0s_index = text.index("c0s")
    r0s = []
    c0s = []
    for ind in range(r0s_index + 1, c0s_index):
        text_split = text[ind].split(',')
        for t in text_split:
            if t.isdigit():
                r0s.append(int(t))
    for ind in range(c0s_index + 1, len(text)):
        text_split = text[ind].split(',')
        for t in text_split:
            if t.isdigit():
                c0s.append(int(t))
    full_frame_r0s = np.array(r0s)
    full_frame_c0s = np.array(c0s)
    return full_frame_r0s, full_frame_c0s
