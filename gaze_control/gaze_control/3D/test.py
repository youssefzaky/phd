import numpy as np
from eye_3D import Eye_Head, quat_rotate, q_id, t_id

s = Eye_Head()
axis = [0, 0, 1]
angle = np.pi/2
sine = np.sin(angle/2)
s.foveate_quat([sine*axis[0], sine*axis[1], sine*axis[2], np.cos(angle/2)])
s.plot()
# s.plot()
# s.head.apply_vel([0, 1, 1], t_id, q_i,d 400)
# print s.head.trans_s
# s.head.apply_vel([0, 1, 0], t_id, q_id, 400)
# s.head.objs[0].apply_vel([0, 0, 1], s.head.trans_s, s.head.quat_s, timesteps=300)
# s.head.objs[0].apply_vel([0, 1, 0], s.head.trans_s, s.head.quat_s, timesteps=300)

# s.head.objs[0].apply_torque([0, 10, 15], s.head.trans_s, s.head.quat_s, timesteps=100)
# s.head.objs[0].apply_torque([0, 0, -5], s.head.trans_s, s.head.quat_s, timesteps=300)
# target = [10, 15, 4]
# s.plot(target=None)
# print s.eye_r.gaze_error_2D(target, s.gaze_trans('r'), s.G_r)
# print s.eye_r.gaze_error_2D(target, s.gaze_trans('l'), s.G_l)
# s.apply_eye_vel([0, 0, 1], 500)
# s.foveate(target)
# target = [10, 15, -4]
# s.foveate_cross(target)
# target = [10, 10, 0]
# s.foveate_cross(target)


# testing if cross product formula leads to half-angle rule
# target = [10, 15, 0]
# eye_r = s.head.objs[1]
# eye_r.apply_vel([0, -1, 0], s.head.trans, q_id, timesteps=700)
# error_2D = eye_r.error_2D(target)
# x = eye_r.r - np.sqrt(np.maximum((eye_r.r ** 2) - (error_2D[0] ** 2) - (error_2D[1] ** 2), 0))
# error_3D = [x, error_2D[0], error_2D[1]]
# e_3D_h = quat_rotate(eye_r.quat.inv() , error_3D)
# gaze = quat_rotate(eye_r.quat, eye_r.bfx)
# axis = -np.cross(gaze, e_3D_h)

# s.plot(vectors=[axis, gaze])

# print gaze, axis
