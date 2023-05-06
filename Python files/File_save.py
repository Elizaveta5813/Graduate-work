import numpy as np

def File_Coordinate(num):
    open('x_{}.txt'.format(num), 'w').close()
    open('y_{}.txt'.format(num), 'w').close()
    open('z_{}.txt'.format(num), 'w').close()

    f_x = open('x_{}.txt'.format(num), 'a')
    f_y = open('y_{}.txt'.format(num), 'a')
    f_z = open('z_{}.txt'.format(num), 'a')

    return f_x, f_y, f_z

def File_Velocity(num):
    open('v_x_{}.txt'.format(num), 'w').close()
    open('v_y_{}.txt'.format(num), 'w').close()
    open('v_z_{}.txt'.format(num), 'w').close()

    v_x_ = open('v_x_{}.txt'.format(num), 'a')
    v_y_ = open('v_y_{}.txt'.format(num), 'a')
    v_z_ = open('v_z_{}.txt'.format(num), 'a')

    return v_x_, v_y_, v_z_

def File_Force(num):
    open('force_x_{}.txt'.format(num), 'w').close()
    open('force_y_{}.txt'.format(num), 'w').close()
    open('force_z_{}.txt'.format(num), 'w').close()

    file_x = open('force_x_{}.txt'.format(num), 'a')
    file_y = open('force_y_{}.txt'.format(num), 'a')
    file_z = open('force_z_{}.txt'.format(num), 'a')

    return file_x, file_y, file_z

def Coordinate_Save(f_x, f_y, f_z,  x, y, z):
    np.savetxt(f_x, x, newline=" ")
    f_x.write("\n")
    np.savetxt(f_y, y, newline=" ")
    f_y.write("\n")
    np.savetxt(f_z, z, newline=" ")
    f_z.write("\n")


def Velocity_Save(v_x_, v_y_, v_z_, V_x, V_y, V_z):
    np.savetxt(v_x_, V_x, newline=" ")
    v_x_.write("\n")
    np.savetxt(v_y_, V_y, newline=" ")
    v_y_.write("\n")
    np.savetxt(v_z_, V_z, newline=" ")
    v_z_.write("\n")

def Force_Save(f_x, f_y, f_z, f0_x, f0_y, f0_z):
    np.savetxt(f_x, f0_x, newline=" ")
    f_x.write("\n")
    np.savetxt(f_y, f0_y, newline=" ")
    f_y.write("\n")
    np.savetxt(f_z, f0_z, newline=" ")
    f_z.write("\n")