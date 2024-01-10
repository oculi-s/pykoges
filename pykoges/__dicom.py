__all__ = ["Dicom"]


def _plt(img3d, nrow, ncol, only):
    import matplotlib.pyplot as plt

    if only:
        plt.imshow(img3d[only], cmap="gray")
        plt.show()
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


def _nearby(seed, dicom, d=10):
    import numpy as np

    x, y, z = seed
    stack = []
    for i in np.arange(x - d / 2, x + d / 2):
        for j in np.arange(y - d / 2, y + d / 2):
            for k in np.arange(z - d / 2, z + d / 2):
                if 0 <= i < dicom.x and 0 <= j < dicom.y and 0 <= k < dicom.z:
                    if _dist((i, j, k), seed) <= np.sqrt(d):
                        stack.append((int(i), int(j), int(k)))
    return stack


def _dist(a, b):
    import numpy as np

    # return np.sqrt((a - b).dot(a - b))
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


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

        self.n = len(dicoms)
        dicoms = [x for x in dicoms if hasattr(x, "SliceLocation")]
        self.shape = [dcm.Columns, len(dicoms), dcm.Rows]
        self.x, self.y, self.z = self.shape
        self.volume = self.x * self.y * self.z

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


class Dicom:
    def __init__(self, datas):
        self.datas = datas

    def __repr__(self):
        from pykoges.utils import arr_to_df_split
        from IPython.display import display

        arr = [[x.id, x.x, x.y, x.z, x.type] for x in self.datas]
        display(arr_to_df_split(arr, column=["코드", "X", "Y(n)", "Z", "type"], n=20))
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

    def plot_axial(self, at=0, nrow=4, ncol=8, only=None):
        _plt(_conv(self.datas[at].data, 2), nrow, ncol, only)

    def plot_sagittal(self, at=0, nrow=4, ncol=8, only=None):
        _plt(_conv(self.datas[at].data, 1), nrow, ncol, only)

    def plot_coronal(self, at=0, nrow=4, ncol=8, only=None):
        _plt(_conv(self.datas[at].data, 0), nrow, ncol, only)

    def upscale(self, dx=2, dy=2, dz=2):
        from scipy.ndimage import zoom
        from tqdm.notebook import tqdm

        res = self.copy()
        for idx, dicom in tqdm(enumerate(res.datas)):
            dicom.data = zoom(dicom.data, (dx, dy, dz))
            dicom.shape = dicom.data.shape
            dicom.x, dicom.y, dicom.z = dicom.shape
            res.datas[idx] = dicom
        return res

    def crop_3d(self, threshold=0.5):
        from scipy import ndimage
        import numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            mask = dicom.data > threshold
            labeled, n = ndimage.label(mask)
            bbox = ndimage.find_objects(labeled == 1)[0]
            dicom.data = np.array(dicom.data[bbox])
            res.datas[idx] = dicom
        return res

    def as4d(self):
        import numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            if len(dicom.shape) != 4:
                dicom.data = np.repeat(dicom.data[..., np.newaxis], 3, -1)
            dicom.data = (dicom.data / np.amax(dicom.data) * 255).astype(np.uint8)
            res.datas[idx] = dicom
        return res

    def point(self, seed_list=[(0, 0)], radius=10):
        res = self.copy().as4d()
        for idx, dicom in enumerate(res.datas):
            seed = seed_list[idx]
            seed = (int(seed[0] * dicom.x), int(seed[1] * dicom.y), int(seed[2]))
            stack = _nearby(seed, dicom, radius)
            for p in stack:
                dicom.data[p[0], p[1], :] = [255, 0, 0]
            res.datas[idx] = dicom
        return res

    def large(self):
        from scipy import ndimage
        import numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            labeled, n = ndimage.label(dicom.data)
            sizes = ndimage.sum(dicom.data, labeled, range(n + 1))
            dicom.data = labeled == np.argmax(sizes)
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

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            dicom.data = dicom.data / np.amax(dicom.data)
            dicom.data = np.where(
                (start <= dicom.data) & (dicom.data <= end), dicom.data, 0
            )
            res.datas[idx] = dicom
        return res

    def fill(self, threshold=0):
        from scipy import ndimage
        import numpy as np

        res = self.copy()
        for idx, dicom in enumerate(res.datas):
            dicom.data = dicom.data / dicom.data.max()
            dicom.data = np.where(
                ndimage.binary_fill_holes(dicom.data > threshold), dicom.data, 0
            ).reshape(dicom.data.shape)
            res.datas[idx] = dicom
        return res

    def seed_growing_3d(self, seed_list=[], threshold=10, radius=10, largest=True):
        import numpy as np
        from scipy import ndimage
        from tqdm.notebook import tqdm

        res = self.copy()
        for idx, dicom in tqdm(enumerate(res.datas)):
            seed = seed_list[idx]
            farm = np.zeros_like(dicom.data)
            seed = (int(seed[0] * dicom.x), int(seed[1] * dicom.y), int(seed[2]))
            stack = _nearby(seed, dicom, radius)
            v = np.mean(dicom.data[stack])
            # dfs bfs
            while stack:
                # for p in stack:
                p = stack.pop()
                x, y, z = p
                if farm[p]:
                    continue
                # if abs(dicom.data[p]-dicom.data[stack].mean()) > threshold:
                #     continue
                if abs(dicom.data[p] - v) * 255 > threshold:
                    # + _dist(p, seed) / dicom.volume
                    continue
                farm[p] = 1
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            if (
                                0 <= x + i < dicom.x
                                and 0 <= y + j < dicom.y
                                and 0 <= z + k < dicom.z
                            ):
                                stack.append((x + i, y + j, z + k))
            # for p in stack:
            #     if (farm[_nearby(p, dicom, 3)] == 0).any():
            #         farm[p] = 0

            farm = ndimage.binary_fill_holes(farm).astype(int)
            if largest:
                labeled, n = ndimage.label(farm)
                result = np.zeros_like(farm)
                result[labeled == labeled[seed]] = 1
                farm = result
            if np.amax(farm) == 0:
                raise ValueError("데이터가 비었습니다.")
            dicom.data = farm
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
