from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
from psbody.meshlite import Mesh


class Perspective_Camera():
    # Input is in numpy array format
    def __init__(self, focal_length_x, focal_length_y, center_x, center_y, trans, axis_angle):
        rotm = cv2.Rodrigues(axis_angle)

        I = np.zeros((3, 3), dtype=np.float32)
        I[0][0] = focal_length_x
        I[1][1] = focal_length_y
        I[2][2] = 1
        I[0][2] = center_x
        I[1][2] = center_y

        E = np.concatenate([rotm[0], trans.reshape(3, 1)], axis=1)

        self.R = tf.constant(np.matmul(I, E), dtype=tf.float32)

    def project(self, points):
        points = tf.concat([points, tf.ones([points.shape[0], 1])], axis=1)
        R = tf.tile(tf.reshape(self.R, [1, 3, 4]), [tf.shape(points)[0], 1, 1])
        points = tf.expand_dims(points, axis=-1)
        res = tf.matmul(R, points)
        res = tf.squeeze(res, axis=-1)
        z = tf.expand_dims(res[:, 2], axis=-1)
        res = tf.divide(res, z)
        return res[:, :2]


if __name__ == '__main__':
    m = Mesh()
    m.load_from_ply('~/Data/HEVA_Validate/S1_Box_1_C1/Res_1/frame0010.ply')
    import cv2

    img = cv2.imread('~/HEVA_Validate/S1_Box_1_C1/Image/frame0010.png')
    import scipy.io as sio

    cam_data = sio.loadmat('~/Data/HEVA_Validate/S1_Box_1_C1/GT/camera.mat', squeeze_me=True, struct_as_record=False)
    cam_data = cam_data['camera']
    cam = Perspective_Camera(cam_data.focal_length[0], cam_data.focal_length[1], cam_data.principal_pt[0],
                             cam_data.principal_pt[1],
                             cam_data.t / 1000.0, cam_data.R_angles)
    v = tf.constant(m.v, dtype=tf.float32)
    j2ds = cam.project(v)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    j2ds = sess.run(j2ds)

    import ipdb

    ipdb.set_trace()
    for p in j2ds:
        x = int(p[0])
        y = int(p[1])
        if x < 0 or y < 0:
            continue
        if x < img.shape[0] and y < img.shape[1]:
            img[x, y, :] = 0
    cv2.imshow('Img', img)
    cv2.waitKey(0)
