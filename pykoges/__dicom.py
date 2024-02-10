__all__ = ["Dicom"]


def _plt(img3d, nrow, ncol, only, dicom):
    import matplotlib.pyplot as plt

    if only:
        plt.title(dicom.id)
        plt.imshow(img3d[only], cmap="gray")
        return

    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(20, 20 * img3d[0].shape[0] / img3d[0].shape[1] * nrow / ncol),
    )
    step = len(img3d) // (ncol * nrow)

    for i in range(nrow):
        for j in range(ncol):
            k = step * (i * ncol + j)
            ax = axs[i, j] if nrow != 1 else axs[j]
            ax.imshow(img3d[k], cmap="gray")
            ax.axis("off")
    plt.suptitle(dicom.id)
    plt.show()


def _conv(d, ax=0):
    import numpy as np

    d = np.array(d)
    return np.array(
        [
            (d[i, :, :].T if ax == 0 else d[:, i, :].T if ax == 1 else d[:, :, i])
            for i in range(d.shape[ax])
        ]
    )


def _nearby(seed, dicom, d=3):
    import numpy as np

    x, y, z = seed
    stack = []
    for i in np.arange(x - d, x + d + 1):
        for j in np.arange(y - d, y + d + 1):
            for k in np.arange(z - d, z + d + 1):
                if 0 <= i < dicom.x and 0 <= j < dicom.y and 0 <= k < dicom.z:
                    if _dist((i, j, k), seed) <= np.sqrt(d):
                        stack.append((int(i), int(j), int(k)))
    return stack


def _dist(a, b):
    import numpy as np

    return max([abs(x) for x in [a[0] - b[0], a[1] - b[1], a[2] - b[2]]])
    # return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _can(img, p, d=3):
    x, y, z = p
    return (
        0
        not in img[
            x - d // 2 : x + d // 2 + 1,
            y - d // 2 : y + d // 2 + 1,
            z - d // 2 : z + d // 2 + 1,
        ]
    )


class DicomDatas:
    def __init__(self, dicoms=[]):
        if not dicoms:
            return
        import numpy as np

        dcm = dicoms[0]
        self.id = dcm.PatientID
        self.name = dcm.PatientName
        self.sex = dcm.PatientSex
        self.birthday = dcm.PatientBirthDate
        self.data = dcm.StudyDate
        self.modality = dcm.Modality
        self.type = dcm.ImageType

        self.dx, self.dz = dcm.PixelSpacing
        self.dy = dcm.SliceThickness
        self.volume = 0

        self.contours = []
        self.areas = []

        self.n = len(dicoms)
        dicoms = [x for x in dicoms if hasattr(x, "SliceLocation")]
        self.shape = [dcm.Columns, len(dicoms), dcm.Rows]
        self.x, self.y, self.z = self.shape

        self.data = np.zeros(self.shape)
        for i, x in enumerate(sorted(dicoms, key=lambda x: x.SliceLocation)):
            self.data[:, i, :] = x.pixel_array.T
        self.data /= self.data.max()

    def __repr__(self):
        from pykoges.utils import arr_to_df
        from IPython.display import display

        arr = [
            ["id", self.id],
            ["x, y, z", (self.x, self.y, self.z)],
        ]
        display(arr_to_df(arr))
        return ""

    def copy(self):
        import copy

        res = self.__class__()
        for k, v in self.__dict__.items():
            try:
                setattr(res, k, copy.deepcopy(v))
            except:
                setattr(res, k, v)
        res.shape = res.data.shape
        res.x, res.y, res.z = list(res.shape)[:3]
        res.volume = res.x * res.y * res.z
        return res

    def as4d(self):
        import numpy as np

        res = self.copy()
        if len(res.shape) != 4:
            res.data = np.repeat(res.data[..., np.newaxis], 3, -1)
        res.data = (res.data / res.data.max() * 255).astype(np.uint8)
        return res

    def point(self, seed, radius=3):
        res = self.copy()
        seed = (
            int(seed[0] * res.x),
            int(seed[1] * res.y),
            # int(seed[2] * dicom.z),
            int(seed[2]),
        )
        stack = _nearby(seed, res, radius)
        for p in stack:
            res.data[p[0], p[1], :] = [0, 255, 0]
        return res

    def plot_axial(self, ncol=8, nrow=4, only=None):
        _plt(_conv(self.data, 2), nrow, ncol, only, self)

    def plot_sagittal(self, ncol=8, nrow=4, only=None):
        _plt(_conv(self.data, 1), nrow, ncol, only, self)

    def plot_coronal(self, ncol=8, nrow=4, only=None):
        _plt(_conv(self.data, 0), nrow, ncol, only, self)


class Dicom:
    def __init__(self, datas):
        self.datas = datas
        self.n = len(datas)

    def __repr__(self):
        from pykoges.utils import arr_to_df_split
        from IPython.display import display

        arr = [[x.id, x.x, x.y, x.z] for x in self.datas]
        display(arr_to_df_split(arr, column=["코드", "X", "Y(n)", "Z"], n=20))
        return ""

    def copy(self):
        datas = [dicom.copy() for dicom in self.datas]
        return Dicom(datas)

    def read(folder_name, n_patient=0, img_type="SRS00001"):
        import zipfile, os, pydicom
        from io import BytesIO
        from tqdm.notebook import tqdm

        if not folder_name:
            raise ValueError("파일을 읽어올 폴더 이름은 필수입력값입니다.")
        if not os.path.exists(folder_name):
            raise ValueError("폴더가 존재하지 않습니다.")
        datas = []
        patients = os.listdir(folder_name)
        if n_patient > 0:
            patients = patients[:n_patient]
        for x in tqdm(patients):
            _dir = os.path.join(folder_name, x)
            dicoms = []
            zf = zipfile.ZipFile(_dir, "r")
            files = [y for y in zf.namelist() if img_type in y]
            for y in files:
                byte = zf.read(y)
                dcm = pydicom.dcmread(BytesIO(byte), force=True)
                dicoms.append(dcm)
            datas.append(DicomDatas(dicoms))
        return Dicom(datas)

    def plot_axial(self, ncol=8, nrow=4, only=None):
        for dicom in self.datas:
            dicom.plot_axial(ncol, nrow, only)

    def plot_sagittal(self, ncol=8, nrow=4, only=None):
        for dicom in self.datas:
            dicom.plot_sagittal(ncol, nrow, only)

    def plot_coronal(self, ncol=8, nrow=4, only=None):
        for dicom in self.datas:
            dicom.plot_coronal(ncol, nrow, only)

    def plot_seed_axial(self, ncol=8, nrow=4, seed_list=[(0, 0, 0)]):
        import matplotlib.pyplot as plt

        fig, axs = plt.figure()
        for dicom in self.datas:
            pass

    def scale(self, dx=2, dy=2, dz=2):
        from scipy.ndimage import zoom
        from tqdm.notebook import tqdm

        res = self.copy()
        for idx, dicom in tqdm(enumerate(res.datas)):
            dicom.data = zoom(dicom.data, (dx, dy, dz))
            dicom.shape = dicom.data.shape
            dicom.x, dicom.y, dicom.z = dicom.shape
            res.datas[idx] = dicom
        return res

    def crop(self, threshold=0.1):
        from scipy import ndimage
        import numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            mask = dicom.data > threshold
            labeled, n = ndimage.label(mask)
            bbox = ndimage.find_objects(labeled == 1)[0]
            dicom.data = np.array(dicom.data[bbox])
            if dicom.data.sum() < 1000:
                res.datas.pop(idx)
            else:
                res.datas[idx] = dicom
        return res.copy()

    def on(self, back, color=None):
        import numpy as np

        front = self.copy().as4d()
        res = back.copy().as4d()
        for idx, dicom in enumerate(res.datas):
            fimg = front.datas[idx].data
            fimg[:, :, :, [1, 2]] = 0
            dicom.data = np.where(fimg, color or fimg, dicom.data)
            res.datas[idx] = dicom
        return res

    def as4d(self):
        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            res.datas[idx] = dicom.as4d()
        return res

    def point(self, seed_list=[(0, 0, 0)], radius=5):
        res = self.copy().as4d()
        for idx, dicom in enumerate(res.datas):
            if idx < len(seed_list):
                seed = seed_list[idx]
            else:
                seed = seed_list[-1]
            res.datas[idx] = dicom.point(seed, radius)
        return res

    def take(self, n=1):
        from scipy import ndimage
        import numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            labeled, k = ndimage.label(dicom.data)
            sizes = ndimage.sum(dicom.data, labeled, range(k + 1))
            labels = np.argsort(sizes)[-n:]
            dicom.data = np.where(np.isin(labeled, labels), dicom.data, 0)
            res.datas[idx] = dicom
        return res

    def range(self, start=0, end=1):
        import numpy as np

        res = self.copy().as4d()
        for idx, dicom in enumerate(res.datas):
            dicom.data = dicom.data / np.amax(dicom.data)
            dicom.data = np.where(
                (start <= dicom.data) & (dicom.data <= end), [1, 0, 0], dicom.data
            )
            res.datas[idx] = dicom
        return res

    def only(self, start=0, end=1):
        import numpy as np
        from tqdm.notebook import tqdm

        res = self.copy()
        for idx, dicom in tqdm(enumerate(res.datas)):
            dicom.data = dicom.data / dicom.data.max()
            dicom.data = np.where(
                (start <= dicom.data) & (dicom.data <= end), dicom.data, 0
            )
            res.datas[idx] = dicom
        return res

    def fill3d(self, threshold=0):
        from scipy import ndimage
        import numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            dicom.data = dicom.data / dicom.data.max()
            dicom.data = np.where(
                ndimage.binary_fill_holes(
                    dicom.data > threshold, structure=np.ones((3, 3, 3))
                ),
                dicom.data,
                0,
            ).reshape(dicom.data.shape)
            res.datas[idx] = dicom
        return res

    def fill(self, threshold=40):
        import cv2, numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            dicom.data = (dicom.data / dicom.data.max() * 255).astype(np.uint8)
            for i in range(dicom.z):
                sliced = dicom.data[:, :, i].copy()
                _, binary = cv2.threshold(sliced, 0, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < threshold:
                        cv2.drawContours(sliced, [contour], 0, 255, -1)
                dicom.data[:, :, i] = sliced
            res.datas[idx] = dicom
        return res

    def drop(self, threshold=40):
        import cv2, numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            dicom.data = (dicom.data / dicom.data.max() * 255).astype(np.uint8)
            for i in range(dicom.z):
                sliced = dicom.data[:, :, i].copy()
                _, binary = cv2.threshold(sliced, 0, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < threshold:
                        cv2.drawContours(sliced, [contour], 0, 0, -1)
                dicom.data[:, :, i] = sliced
            res.datas[idx] = dicom
        return res

    def seed_growing(self, seed_list=[], threshold=10, kernel_size=1):
        import numpy as np
        from tqdm.notebook import tqdm

        res = self.copy()
        for idx, dicom in tqdm(enumerate(res.datas)):
            dicom.data = (dicom.data / dicom.data.max() * 255).astype(np.uint8)
            if idx < len(seed_list):
                seed = seed_list[idx]
            else:
                seed = seed_list[-1]
            farm = np.zeros_like(dicom.data)
            seed = (
                int(seed[0] * dicom.x),
                int(seed[1] * dicom.y),
                # int(seed[2] * dicom.z),
                int(seed[2]),
            )
            if not dicom.data[seed]:
                continue
            stack = [seed]
            # dfs bfs 선택가능
            while stack:
                x, y, z = p = stack.pop()
                farm[p] = 1
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            p = (x + i, y + j, z + k)
                            if (
                                0 <= x + i < dicom.x
                                and 0 <= y + j < dicom.y
                                and 0 <= z + k < dicom.z
                                and not farm[p]
                                and _can(dicom.data, p, kernel_size)
                                and abs(dicom.data[p] - dicom.data[seed]) <= threshold
                            ):
                                stack.append(p)
            if farm.max() == 0:
                raise ValueError("데이터가 비었습니다.")
            dicom.data = farm
            res.datas[idx] = dicom
        return res

    def dilate(self, kernal_size=5):
        import cv2, numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            dicom.volume = 0
            dicom.contours = []
            dicom.areas = []
            for i in range(dicom.z):
                sliced = dicom.data[:, :, i]
                contours = cv2.findContours(
                    sliced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = contours[0] if len(contours) == 2 else contours[1]
                smoothed = np.zeros_like(sliced)
                for contour in contours:
                    peri = cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, 0.001 * peri, True)

                    cv2.drawContours(smoothed, [contour], 0, 255, -1)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_DILATE, (kernal_size, kernal_size)
                )
                dilate = cv2.morphologyEx(smoothed, cv2.MORPH_DILATE, kernel)
                contours = cv2.findContours(
                    dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = contours[0] if len(contours) == 2 else contours[1]
                area = sum([cv2.contourArea(contour) for contour in contours])
                # dilate로 늘어난 면적의 90%만 가져옴
                dicom.data[:, :, i] = dilate
                dicom.contours.append(contours)
                dicom.areas.append(area)
                dicom.volume += area * 0.9 * (dicom.dz * dicom.dx * dicom.dy)
            res.datas[idx] = dicom
        return res

    # def seed_growing_2d(img, seed, threshold=60, largest=True):
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     from scipy import ndimage

    #     res = np.zeros_like(img)
    #     stack = [seed]

    #     while stack:
    #         x, y = stack.pop()
    #         if res[x, y] == 0 and abs(img[x, y] - img[seed]) < threshold:
    #             res[x, y] = 255
    #             for i in range(-1, 2):
    #                 for j in range(-1, 2):
    #                     if 0 <= x + i < img.shape[0] and 0 <= y + j < img.shape[1]:
    #                         stack.append((x + i, y + j))
    #     img = ndimage.binary_fill_holes(res).astype(int)
    #     if largest:
    #         res, num_labels = ndimage.label(img)
    #         sizes = ndimage.sum(img, res, range(num_labels + 1))
    #         max_label = np.argmax(sizes)
    #         img = res == max_label
    #     plt.imshow(img)
    #     return img

    def smooth(self, kernal_size=3):
        import cv2, numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            for i in range(dicom.z):
                sliced = dicom.data[:, :, i]
                contours = cv2.findContours(
                    sliced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = contours[0] if len(contours) == 2 else contours[1]
                smoothed = np.zeros_like(sliced)
                for contour in contours:
                    peri = cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, 0.001 * peri, True)

                    cv2.drawContours(smoothed, [contour], 0, 255, -1)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_OPEN, (kernal_size, kernal_size)
                )
                dilate = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
                dicom.data[:, :, i] = np.where(dilate, sliced, 0)
            res.datas[idx] = dicom
        return res

    def kmeans(self, K=4):
        import cv2, numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            vectorized = dicom.data.reshape((-1, 1)).astype(np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            attempts = 10
            ret, label, center = cv2.kmeans(
                vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
            )
            label = label.flatten()

            center = np.uint8(center)
            center[label[0]] = 0
            dicom.data = center[label].reshape(dicom.data.shape)
            res.datas[idx] = dicom
        return res
