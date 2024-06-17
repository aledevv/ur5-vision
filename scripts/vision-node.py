import rospy
from vision.msg import block
from pathlib import Path
import sys
import os
import copy
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import Block_detection as Lego
import Region_of_interest as roi
import math
from tf.transformations import quaternion_from_euler
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time

# --------------- DIRECTORIES ---------------
ROOT = Path(__file__).resolve().parents[1]  # vision directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.abspath(ROOT))

ZED_IMG = str(ROOT) + '/images/img_ZED_cam.png'
ROI_IMG = str(ROOT) + '/images/ROI_table.png'
LINE_IMG = str(ROOT) + '/images/line-height.png'
bridge = CvBridge()

# --------------- WORLD PARAMS ---------------

R_cloud_to_world = np.matrix([[0, -0.49948, 0.86632], [-1., 0., 0.], [0., -0.86632, -0.49948]])
x_camera = np.array([-0.9, 0.24, -0.35])
base_offset = np.array([0.5, 0.35, 1.75])
block_offset = [0.0189, -0.007, 0]
voxel_size = 0.004


TABLE_OFFSET = 0.86 + 0.1

# --------------- FLAGS e GLOBALS ---------------
can_acquire_img = True
can_take_point_cloud = False
send_next_msg = True
block_list = []
measures = 0        # num of measures for world position of blocks
icp_threshold = 0.0001

class Point:
    # @Description Class to store points in the world space useful to estimate block pose
    # @Fields: coordinates x, y, z and pixels coordinates in ROI image

    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.px = ()

    def set(self, x, y, z, px):
        # @Description  Method to all parameters in once

        self.x = x
        self.y = y
        self.z = z
        self.px = px

    def is_min_x(self, coords, px):
        # @Description Compares this point with another to find the one with lower value of x
        # @Parameters tuple of world coordinates of second point and pixels coordinates of this latter

        if coords[0] < self.x:  # checks if x of 2nd point is lower than the one stored by this one,  if it does updates data
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
            self.px = px

    def is_min_y(self, coords, px):
        # @Description Compares this point with another to find the one with lower value of y
        # @Parameters tuple of world coordinates of second point and pixels coordinates of this latter

        if coords[1] < self.y:  # checks if y of other point is lower than y of this object, if it does updates data
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
            self.px = px

    def is_max_y(self, coords, px):
        # @Description Compares this point with another to find the one with greater value of y
        # @Parameters tuple of world coordinates of second point and pixels coordinates of this latter

        if coords[1] > self.y:  # checks whether y of other point is greater than this object, if it does updates data
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
            self.px = px

    def info(self):
        # @Description Prints Point info
        print("x,y,z, PX: ", self.x, self.y, self.z, self.px)


def block_info(block):
    # @Description Prints Block info

    print("MESSAGE: ")
    print("\tlabel: " + str(block.label))
    print("\tx: " + str(block.x))
    print("\ty: " + str(block.y))
    print("\tz: " + str(block.z))
    print("\troll: " + str(block.roll))
    print("\tpitch: " + str(block.pitch))
    print("\tyaw: " + str(block.yaw))


def get_img(img):
    # @Description Callback function to store image from zed node and start detection process
    # @Parameters Image from Zed camera

    global can_acquire_img
    global can_take_point_cloud
    global block_list

    if can_acquire_img:
        try:
            cv_obj = bridge.imgmsg_to_cv2(img, "bgr8")      # convert image to cv2 obj
        except CvBridgeError as e:
            print(e)

        cv.imwrite(ZED_IMG, cv_obj)

        # FIND ROI
        roi.find_roi(ZED_IMG)           # find ROI

        # Detection phase
        block_list = Lego.detection(ROI_IMG)    # get list of detected blocks

        # Copying ROI image to draw line for pose estimations
        img = cv.imread(ROI_IMG, cv.IMREAD_COLOR)
        cv.imwrite(LINE_IMG, img)

        # FLAG TO CHANGE
        can_acquire_img = False
        can_take_point_cloud = True

def get_point_cloud(point_cloud):

    # @Description Callback function to collect point cloud and estimate center and pose of each block detected
    # @Parameters point cloud from zed node

    global can_take_point_cloud
    global block_list
    global measures

    if can_take_point_cloud:

        for block in block_list:
            # finding x, y, z of the center of the block leveraging center of the bounding boxes
            for data in point_cloud2.read_points(point_cloud, field_names=['x', 'y', 'z'], skip_nans=True, uvs=[block.center]):
                block.point_cloud_coord = [data[0], data[1], data[2]]

            # computing world coordinates through a transformation matrix and correcting result adding offsets
            block.world_coord = R_cloud_to_world.dot(block.point_cloud_coord) + x_camera + base_offset + block_offset
            block.world_coord[0, 2] = 0.86999       # z of the block is a constant

        # make 3 measures of block world coordinates
        if measures < 3:
            measures+=1
        else:
            can_take_point_cloud = False
            measures = 0

            # calculate the pose of the block
            for block in block_list:
                block.yaw = find_pose(point_cloud, block)

            # print block info
            Lego.print_block_info(block_list)

            # publish details to motion node
            msg_pub(block_list)


def create_open3d_point_cloud(point_cloud_box):
    # Creating an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Converting the list parameter in numpy array (Nx3)
    points = np.array(point_cloud_box)

    # Add points in list to point cloud
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


# Visualization of the point cloud
def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])


# Loading Mesh model of the block having a specific label
def load_mesh_model(block_label):
    mesh = o3d.io.read_triangle_mesh('models/'+block_label+'/mesh/'+block_label+'.stl').sample_points_poisson_disk(6138)

    #o3d.visualization.draw_geometries([mesh])

    return mesh


def execute_icp(source, target, threshold):
    # Setting convergence criteria
    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000)

    # ICP algorithm
    transformation_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        icp_criteria)

    return transformation_icp.transformation


def extract_rpy_from_transformation(transformation):
    # Extracting 3x3 rotation matix from the 4x4 one
    rotation_matrix = transformation[:3, :3].copy()

    # Use scipy to convert rotation matrix to RPY angles
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz')

    if roll > 3.14:
        roll = roll - 3.14
    if pitch > 3.14:
        pitch = pitch - 3.14
    if yaw > 3.14:
        yaw = yaw - 3.14

    return roll, pitch, yaw


def euler_to_rotation_matrix(euler_angles):
    """
    Calcola la matrice di rotazione a partire da angoli di Eulero.

    Args:
    - euler_angles (list or np.ndarray): Angoli di Eulero [x, y, z] in radianti.

    Returns:
    - numpy.ndarray: Matrice di rotazione 3x3.
    """
    # Estrai gli angoli di Eulero lungo gli assi x, y, z
    angle_x, angle_y, angle_z = euler_angles

    # Calcola le matrici di rotazione per ciascun asse
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle_x), -np.sin(angle_x)],
                           [0, np.sin(angle_x), np.cos(angle_x)]])

    rotation_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                           [0, 1, 0],
                           [-np.sin(angle_y), 0, np.cos(angle_y)]])

    rotation_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                           [np.sin(angle_z), np.cos(angle_z), 0],
                           [0, 0, 1]])

    # Calcola la matrice di rotazione totale (rotazione intorno a z, poi y, poi x)
    rotation_matrix = rotation_z @ rotation_y @ rotation_x

    return rotation_matrix


def correct_pc_orientation(pcd):
    # Definisci l'asse di rotazione (per esempio attorno all'asse y)

    global base_offset

    translation_vector = [0, 0, 0.86992]

    # Inizializza una matrice identità 4x4
    translation_matrix = np.eye(4)

    # Imposta la parte di traslazione della matrice
    translation_matrix[0:3, 3] = translation_vector

    # Applica la trasformazione alla nuvola di punti
    pcd.transform(translation_matrix)

    return pcd


def draw_registration_result(source, target, transformation):
    source_tmp = copy.deepcopy(source)
    target_tmp = copy.deepcopy(target)
    source_tmp.paint_uniform_color([1, 0.706, 0])
    target_tmp.paint_uniform_color([0, 0.651, 0.929])
    source_tmp.transform(transformation)
    o3d.visualization.draw_geometries([source_tmp, target_tmp])


def prepare_point_clouds(source, target, voxel_size):
    trans_init = np.array([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    #target = correct_pc_orientation(target)
    target.transform(trans_init)

    #draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh


def preprocess_point_cloud(pcd, voxel_size):
    #downsampling with voxel size
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5

    #Computing FPFH feature with search radius feature
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                               o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                                                                                    max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print("EXECUTING RANSAC on downsampled point clouds")

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999)
    )

    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print("PERFORMING Point-to-plane ICP to refine the pose estimation")

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return result


def rotate_point_cloud(pcd, rotation_vector):
    """
    Ruota la nuvola di punti secondo il vettore di rotazione specificato.

    Args:
        pcd (open3d.geometry.PointCloud): La nuvola di punti da ruotare.
        rotation_vector (tuple): Un vettore di 3 angoli in radianti (roll, pitch, yaw).

    Returns:
        open3d.geometry.PointCloud: La nuvola di punti ruotata.
    """
    # Crea la matrice di rotazione dalle componenti del vettore
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(rotation_vector)
    # Applica la rotazione alla nuvola di punti
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    return pcd


# Operation to get and elaborate point cloud of blocks
def get_point_cloud2(point_cloud):
    # @Description Callback function to collect point cloud and estimate center and pose of each block detected
    # @Parameters point cloud from zed node

    global can_take_point_cloud
    global block_list
    global measures
    global icp_threshold
    global voxel_size

    if can_take_point_cloud:

        point_cloud_box = []

        if len(block_list) == 0:
            print("NO BLOCK DETECTED")
            sys.exit(1)

        for block in block_list:
            for x in range(int(block.x1), int(block.x2)):
                for y in range(int(block.y1), int(block.y2)):
                    for point in point_cloud2.read_points(point_cloud, field_names=['x', 'y', 'z'], skip_nans=True, uvs=[(int(x), int(y))]):
                        point_cloud_box.append(np.array(point))
                        block.point_cloud_coord = [point[0], point[1], point[2]]

            # computing world coordinates through a transformation matrix and correcting result adding offsets
            block.world_coord = R_cloud_to_world.dot(block.point_cloud_coord) + x_camera + base_offset + block_offset
            block.world_coord[0, 2] = 0.86999  # z of the block is a constant
            #print("WORLD:", block.world_coord)

            target = create_open3d_point_cloud(point_cloud_box)

            rotate_point_cloud(target, [0, 0, 0])

            print("LABEL: ", block.label)
            #visualize_point_cloud(pcd)

            source = load_mesh_model(block.label)

            #correct_pc_orientation(source)

            #draw_registration_result(source, target, np.identity(4))

            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_point_clouds(source, target, voxel_size)

            result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

            print("RANSAC result: ", result_ransac)

            result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, result_ransac)

            print("ICP result: ", result_icp)

            draw_registration_result(source_down, target_down, result_icp.transformation)


            #draw_registration_result(source, target, transformation)

            #print("Transformation matrix:\n", transformation)

            roll, pitch, yaw = extract_rpy_from_transformation(result_icp.transformation)

            print("Roll: ", roll)
            print("Pitch: ", pitch)
            print("Yaw: ", yaw)

        can_take_point_cloud = False

        msg_pub(block_list)


        #     # finding x, y, z of the center of the block leveraging center of the bounding boxes
        #     for data in point_cloud2.read_points(point_cloud, field_names=['x', 'y', 'z'], skip_nans=True,
        #                                          uvs=[block.center]):
        #         block.point_cloud_coord = [data[0], data[1], data[2]]
        #
        #     # computing world coordinates through a transformation matrix and correcting result adding offsets
        #     block.world_coord = R_cloud_to_world.dot(block.point_cloud_coord) + x_camera + base_offset + block_offset
        #     block.world_coord[0, 2] = 0.86999  # z of the block is a constant
        #
        # # make 3 measures of block world coordinates
        # if measures < 3:
        #     measures += 1
        # else:
        #     can_take_point_cloud = False
        #     measures = 0
        #
        #     # calculate the pose of the block
        #     for block in block_list:
        #         block.yaw = find_pose(point_cloud, block)
        #
        #     # print block info
        #     Lego.print_block_info(block_list)
        #
        #     # publish details to motion node
        #     msg_pub(block_list)

def find_pose(point_cloud, block):
    # @Description Function to compute object pose. Basically, the block is "sliced" at a specific height to find the 2
    #   useful points:
    #   - x_min: the nearest point the camera
    #   - y_min: the rightmost point (near the arm)
    #   these points are used to calculate yaw angle
    # @Parameters point cloud and the block
    # @Returns yaw angle (radiant)

    # CONSTANTS (used only here)
    table_height = 0.88
    target_height = 0.012 + table_height

    selected_points = []       # list of points at the target_height value

    print("Finding pose...")

    min_x = Point()
    min_y = Point()
    # max_y = Point()

    # scan point cloud of the bounding box
    for x in range(int(block.x1), int(block.x2)):
        for y in range(int(block.y1), int(block.y2)):
            for data in point_cloud2.read_points(point_cloud, field_names=['x', 'y', 'z'], skip_nans=True, uvs=[(x,y)]):
                point_coords = [data[0], data[1], data[2]]

                # transform current point to world coordinate
                point_world = R_cloud_to_world.dot(point_coords) + x_camera + base_offset

                # check whether it belongs to target height
                if abs(point_world[0, 2] - target_height) <= 0.001:

                    current_coords = (point_world[0, 0], point_world[0, 1], point_world[0, 2])
                    selected_points.append(current_coords)

                    if len(selected_points) == 1:   # if it is the first point store it as the min_x, min_y
                        color_pixel(x, y, 'red')
                        min_x.set(point_world[0, 0], point_world[0, 1], point_world[0, 2], (x, y))
                        min_y.set(point_world[0, 0], point_world[0, 1], point_world[0, 2], (x, y))
                        # max_y.set(point_world[0, 0], point_world[0, 1], point_world[0, 2], (x, y))
                    else:
                        color_pixel(x, y, 'red')
                        min_x.is_min_x(current_coords, (x, y))
                        min_y.is_min_y(current_coords, (x, y))
                        # max_y.is_max_y(current_coords, (x, y))

    # Print 3 vertices info
    #min_x.info()
    #min_y.info()
    #max_y.info()

    # Show in yellow points of interest
    color_pixel(min_x.px[0], min_x.px[1], 'yellow')
    color_pixel(min_y.px[0], min_y.px[1], 'yellow')
    # color_pixel(max_y.px[0], max_y.px[1], 'yellow')

    # Finding yaw angle
    yaw = math.atan2(min_y.x - min_x.x, min_x.y - min_y.y)

    return yaw


def color_pixel(x, y, color):
    # @Description function to color pixels in a image
    # @Parameters x and y coordinaets of pixels to color and a string indicating to the color to use

    img = cv.imread(LINE_IMG, cv.IMREAD_COLOR)

    if color == 'red':
        img[y][x] = np.array([0, 0, 255])
    elif color == 'yellow':
        img[y][x] = np.array([0, 255, 255])

    cv.imwrite(LINE_IMG, img);


def to_quaternions(r, p ,y):
    # @Description function transform RPY angles to Quaternions
    # @Parameters RPY angles
    # @Returns Quaternion

    return quaternion_from_euler(r, p, y)


def msg_pub(block_list):
    # @Description function that prepares and sends a message to motion node
    # @Parameters list of detected blocks

    global send_next_msg

    if send_next_msg:

        msg = block()
        if len(block_list) > 0:
            current_block = block_list.pop()
        if len(block_list) == 0:
            print("PUBLISHED ALL BLOCKS")
            send_next_msg = False
            return

        # Preparing msg
        msg.label = current_block.label
        msg.x = round(current_block.world_coord[0, 0], 6)
        msg.y = round(current_block.world_coord[0, 1], 6)
        msg.z = round(current_block.world_coord[0, 2], 6)
        msg.roll= 0.0
        msg.pitch = 0.0
        msg.yaw = current_block.yaw

        # QUATERNION conversion
        q = to_quaternions(msg.roll, msg.pitch, msg.yaw)

        #block_info(msg)    # print msg info

        pub.publish(msg)
        rate.sleep()

        send_next_msg = False
        print("Waiting for sending next block")

# ----------------------------------------------- MAIN -----------------------------------------------


if __name__ == '__main__':

    # Publishers
    pub = rospy.Publisher('vision/position', block, queue_size=1)          # Vision msg publisher

    # Subscribers
    img_sub = rospy.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, get_img)
    point_cloud_sub = rospy.Subscriber("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2, get_point_cloud2, queue_size=1)   # Subscriber Point cloud

    rospy.init_node('block_detector', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
