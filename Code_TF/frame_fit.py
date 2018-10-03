from __future__ import print_function

import os

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import tensorflow as tf
from functools import partial
from psbody.meshlite import Mesh
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt

import util
from smpl_batch import SMPL
from util import Counter


def visualize_kpt(j2d_est, imgs, j2ds, counter, save_dir, save_interval=3):
    counter.count()
    if counter.c % save_interval != 0:
        return

    _, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), dpi=100)
    for idx in range(0, util.NUM_VIEW):
        x_est = [float(j2d[1]) for j2d in j2d_est[idx]]
        y_est = [float(j2d[0]) for j2d in j2d_est[idx]]
        x_gt = [float(j2d[1]) for j2d in j2ds[idx]]
        y_gt = [float(j2d[0]) for j2d in j2ds[idx]]
        idx_c, idx_r = idx // 2, idx % 2
        ax[idx_c][idx_r].imshow(imgs[idx])
        ax[idx_c][idx_r].scatter(x=y_est, y=x_est, s=10, c='b', marker='o')
        ax[idx_c][idx_r].scatter(x=y_gt, y=x_gt, s=10, c='r', marker='o')

    plt.savefig(os.path.join(save_dir, "{:03d}.png".format(counter.c)))
    plt.cla()


def regress_smpl_param(img_path):
    print("img_path:", img_path)

    imgs, j2ds, segs, cams = util.load_data(img_path, util.NUM_VIEW)

    # imgs: list length 3, each numpy array (480, 640, 3)
    # j2ds: np array (3, 14, 2)
    # segs: [None, None, None]
    # cams: list length 3, each camera.Perspective_Camera object

    # embed(header='after load data')

    # j2ds = tf.constant(j2ds, dtype=tf.float32)
    initial_param, pose_mean, pose_covariance = util.load_initial_param()

    # initial_param: np array shape(85,)
    # pose_mean: np array shape (69,)
    # pose_covariance: np array shape (69, 69)

    # embed(header='after load init param')

    pose_mean = tf.constant(pose_mean, dtype=tf.float32)
    pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)

    param_shape = tf.Variable(initial_param[:10].reshape([1, -1]), dtype=tf.float32)
    param_rot = tf.Variable(initial_param[10:13].reshape([1, -1]), dtype=tf.float32)
    param_pose = tf.Variable(initial_param[13:82].reshape([1, -1]), dtype=tf.float32)
    param_trans = tf.Variable(initial_param[-3:].reshape([1, -1]), dtype=tf.float32)
    param = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)

    smpl_model = SMPL(util.SMPL_PATH)
    j3ds, v = smpl_model.get_3d_joints(param, util.SMPL_JOINT_IDS)
    j3ds = tf.reshape(j3ds, [-1, 3])

    j2ds_est = [cams[idx].project(tf.squeeze(j3ds)) for idx in range(0, util.NUM_VIEW)]
    j2ds_est = tf.convert_to_tensor(j2ds_est)

    objs = {}
    for idx in range(0, util.NUM_VIEW):
        for j, jdx in enumerate(util.TORSO_IDS):
            objs['J2D_%d_%d' % (idx, j)] = tf.reduce_sum(tf.square(j2ds_est[idx][jdx] - j2ds[idx][jdx]))
    loss = tf.reduce_mean(objs.values())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if util.VIS_OR_NOT:
        save_dir = './optimize_vis_1'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        func_lc = partial(visualize_kpt, imgs=imgs, j2ds=j2ds, save_dir=save_dir, counter=Counter())
    else:
        func_lc = None
    optimizer = scipy_pt(loss=loss, var_list=[param_rot, param_trans],
                         options={'ftol': 0.001, 'maxiter': 500, 'disp': True}, method='L-BFGS-B')
    optimizer.minimize(sess, fetches=[j2ds_est], loss_callback=func_lc)

    objs = {}
    pose_diff = tf.reshape(param_pose - pose_mean, [1, -1])
    objs['J2D_Loss'] = tf.reduce_sum(tf.square(j2ds_est - j2ds))
    objs['Prior_Loss'] = 5 * tf.squeeze(tf.matmul(tf.matmul(pose_diff, pose_covariance), tf.transpose(pose_diff)))
    objs['Prior_Shape'] = 5 * tf.squeeze(tf.reduce_sum(tf.square(param_shape)))

    loss = tf.reduce_mean(objs.values())
    optimizer = scipy_pt(loss=loss, var_list=[param_rot, param_trans, param_pose, param_shape],
                         options={'ftol': 0.001, 'maxiter': 500, 'disp': True}, method='L-BFGS-B')
    if util.VIS_OR_NOT:
        save_dir = './optimize_vis_2'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        func_lc = partial(visualize_kpt, imgs=imgs, j2ds=j2ds, save_dir=save_dir, counter=Counter())
    else:
        func_lc = None
    optimizer.minimize(sess, fetches=[j2ds_est], loss_callback=func_lc)

    v_final = sess.run(v)
    model_f = sess.run(smpl_model.f)
    model_f = model_f.astype(int).tolist()
    pose_final, betas_final, trans_final = sess.run(
        [tf.concat([param_rot, param_pose], axis=1), param_shape, param_trans])

    m = Mesh(v=np.squeeze(v_final), f=model_f)
    out_ply_path = img_path.replace('Image', 'Res_1')
    extension = os.path.splitext(out_ply_path)[1]
    out_ply_path = out_ply_path.replace(extension, '.ply')
    m.write_ply(out_ply_path)

    res = {'pose': pose_final, 'betas': betas_final, 'trans': trans_final}
    out_pkl_path = out_ply_path.replace('.ply', '.pkl')
    with open(out_pkl_path, 'wb') as fout:
        pkl.dump(res, fout)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_prefix', type=str)
    parser.add_argument('index', type=int)
    args = parser.parse_args()

    print("args:")
    print("data_prefix: {}".format(args.data_prefix))
    print("index: {}".format(args.index))

    img_files = glob.glob(os.path.join(util.HEVA_PATH, args.data_prefix + '_1_C1', 'Image', '*.png'))
    print("get nr img file: ", len(img_files))
    regress_smpl_param(img_files[args.index])


if __name__ == '__main__':
    main()
