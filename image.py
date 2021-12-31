import numpy as np
from scipy.spatial.transform.rotation import Rotation


class Image:
    uri: str
    campos: (float, float, float)  # [x,y,z]
    camrpy: (float, float, float)  # [r,p,y]
    fov: float
    # distortion: float  # This comes later

    height: int
    width: int

    def __init__(self, config):
        print(config)

        def clean_geom(string: str):
            string = string.replace("[", '').replace("]", "")
            return [float(v) for v in string.split(" ")]

        self.uri = config.get("file_name")
        self.campos = clean_geom(config.get("wkt_geom"))
        self.vppos = clean_geom(config.get("vp_geom"))
        self.camrpy = [float(config.get(v)) for v in ["roll", "pitch", "yaw"]]
        self.width = int(config.get("x_pixels"))
        self.height = int(config.get("y_pixels"))
        self.fov = float(config.get("fov"))

        print(self.__dict__)

    def rs_matrix(self):
        r, p, y = self.camrpy
        print("HEre:", r, p, y)
        fe = lambda a, x: Rotation.from_euler(a, x, degrees=True).as_matrix()
        rm = fe('x', 90 - p)  # Generate rx - Rotation from pitch
        rm = np.matmul(rm, fe('y', -r))  # Generate ry - Rotation from roll
        rm = np.matmul(rm, fe('z', y - 180))  # Generate rz - Rotation from yaw

        # rm = np.matmul(rm, self.campos)

        return rm

    @property
    def aspect(self):
        return self.width / self.height
