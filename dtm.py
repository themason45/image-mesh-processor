import matplotlib.pyplot as plt
from osgeo import gdal, ogr
import numpy as np
import trimesh
from pyproj import transform, Proj

from trimesh.parent import Geometry3D
# from trimesh.ray.ray_triangle import RayMeshIntersector
from trimesh.viewer import SceneViewer

from embreeintersector import RayMeshIntersector
from image import Image

P_EPSG4326 = Proj("epsg:4326")
P_EPSG4978 = Proj("epsg:4978")


class Dtm:
    image: Image
    dtm_url: str
    viewDistance: float = 1.0

    @staticmethod
    def convert_coords_to_xy(lon, lat, z):
        return *transform(P_EPSG4326, P_EPSG4978, lon, lat), z

    @staticmethod
    def convert_xy_to_coords(x, y, z):
        return *transform(P_EPSG4978, P_EPSG4326, x, y), z

    @staticmethod
    def to_pixelcoords(val, offset, ratio):
        return int((val - offset) / ratio)

    @staticmethod
    def extent_to_offset(xmin, xmax, ymin, ymax, gt):
        p1 = (xmin, ymax)
        p2 = (xmax, ymin)

        print(gt)

        row1 = int((p1[1] - gt[3]) / gt[5])
        col1 = int((p1[0] - gt[0]) / gt[1])

        row2 = int((p2[1] - gt[3]) / gt[5])
        col2 = int((p2[0] - gt[0]) / gt[1])

        return col1, row1, int(np.abs(col2 - col1)), int(np.abs(row2 - row1))

    @staticmethod
    def get_nearby_hgt(lon, lat):
        names = []
        for i in range(-1, 2):
            la = int(lat) + i
            for j in range(-1, 2):
                ln = int(lon) + j
                las = "{0}{1:02d}".format('N' if la > 0 else 'S', int(np.abs(la)))
                lns = "{0}{1:03d}".format('E' if ln > 0 else 'W', int(np.abs(ln)))

                names.append(f"{las}{lns}.hgt")

        return names

    def process_dtm(self):
        # Calculate bounds of camera viewport

        # First load the correct HGT file

        names = Dtm.get_nearby_hgt(*(np.round(x) for x in self.image.campos[:2]))
        print(names[4])

        ds = gdal.Open(f"ukhgt/{names[4]}", gdal.GA_ReadOnly)  # type: gdal.Dataset
        gt = ds.GetGeoTransform()
        rb = ds.GetRasterBand(1)  # type: gdal.Band
        print(rb.ComputeRasterMinMax())

        # Generate a bounding rectangle for the camera position
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(*self.image.campos)

        bounds = point.Buffer(0.1)  # type: ogr.Geometry
        envel = bounds.GetEnvelope()
        # Crop heightmap to within 0.1º of the camera location
        img_array = np.array(rb.ReadAsArray(*Dtm.extent_to_offset(*envel, gt)))

        # plt.imshow(img_array)
        # plt.show()

        vertices = np.empty((np.prod(img_array.shape)), dtype=np.ndarray)
        faces = np.empty(((img_array.shape[0] - 1) * (img_array.shape[1] - 1)), dtype=np.ndarray)

        i = 0
        for y in range(0, img_array.shape[1]):
            for x in range(0, img_array.shape[0]):
                hm_coords = (img_array.shape[0] - 1 - y, x)
                vertices[i] = np.array([x, y, img_array[hm_coords]])
                i += 1

        j = 0
        for y in range(0, img_array.shape[0] - 1):
            for x in range(0, img_array.shape[1] - 1):
                p1 = (y * img_array.shape[1] + x)
                p2 = p1 + 1
                p3 = ((y + 1) * img_array.shape[1] + x)

                faces[j] = np.array([p1, p2, p3])
                j += 1

        mesh = trimesh.Trimesh(vertices.tolist(), faces.tolist())
        mesh.export("export.obj", "obj")
        print(mesh.extents)

        # x, y = self.image.campos[:2]
        # tpc = Dtm.to_pixelcoords
        # pt = (tpc(x, gt[3], gt[5]) - tpc(envel[1], gt[3], gt[5]),
        #       tpc(y, gt[0], gt[1]) - tpc(envel[3], gt[0], gt[1]),
        #       0)

        scene = mesh.scene()

        img = self.image

        cx, cy = int(img_array.shape[0] / 2), int(img_array.shape[1] / 2)

        sf = np.abs(np.subtract(*rb.ComputeRasterMinMax())) / mesh.extents[2]

        camera_altitude = (img.campos[2] - img_array[cy, cx]) / sf
        pt = (cx, cy, camera_altitude)
        print(pt)

        camera = trimesh.scene.Camera(name="Camera", resolution=(img.width / 40, img.height / 40),
                                      fov=(img.fov, img.fov))
        # rm = np.empty((4, 4))
        # rm[3, 3] = 1
        # rm[:3, :3] = img.rs_matrix()
        ro, pi, ya = img.camrpy
        print(ro, pi, ya)
        rm = trimesh.transformations.euler_matrix(180 + ro, pi, -ya, axes="sxyz")

        from trimesh.creation import camera_marker

        cammeshes = camera_marker(camera, marker_height=50)  # type: [Geometry3D]

        for cammesh in cammeshes:
            cammesh.apply_transform(-rm)
            cammesh.apply_translation(pt)

            scene.add_geometry(cammesh)

        vectors, origins = camera.to_rays()
        print(vectors)

        origins = np.tile(pt, (vectors.shape[0], 1))

        _rm = trimesh.transformations.euler_matrix(ro, 90 + pi, ya, axes="sxyz")[:3, :3]
        print(_rm)
        for i, v in enumerate(vectors):
            vectors[i] = np.matmul(v, _rm)
        print(vectors)
        # ro, rd = np.array([pt]), np.array([[np.cos(ya) * np.cos(pi), np.sin(ya) * np.cos(pi), np.sin(pi)]])
        rmi = RayMeshIntersector(mesh)
        locations, index_ray, index_tri = rmi.intersects_location(origins, vectors, multiple_hits=False)

        with open("out.xyz", "w") as f:
            for row in locations:
                f.write(",".join([str(v) for v in row]))
                f.write("\n")

        scene.add_geometry(trimesh.points.PointCloud(locations))

        # viewer = SceneViewer(scene)
        # viewer.toggle_grid()
        # viewer.toggle_axis()
        # viewer.show()

        # print(img_array)
        # plt.imshow(img_array)
        # plt.show()
