import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sciann as sn
from sciann.utils.math import diff, sqrt, abs
import tensorflow as tf
from itertools import cycle

# Input part
x = sn.Variable('x', dtype='float64')
y = sn.Variable('y', dtype='float64')

E = sn.Variable('E', dtype='float64')
K = sn.Variable('K', dtype='float64')
nu = 0.3
Opt = 'adam'#'adam'  # or 'scipy-l-BFGS-B'
# Thermal Part
layers = 5*[40]
Tn      = sn.Functional('Tn',     [x,y], layers, 'tanh')

T = Tn*x*(x-1.0) + 1.0-x
q_x_O  = sn.Functional('gT_x',  [x,y], layers, 'tanh')
q_y_O  = sn.Functional('gT_y',  [x,y], layers, 'tanh')

gr_T_x = diff(T, x)
gr_T_y = diff(T, y)
q_x    = (-K)*(gr_T_x)
q_y    = (-K)*(gr_T_y)
act = 'tanh'
# Mechanical part
# Mechanical part
u_xn       = sn.Functional('u_xn', [x,y], layers, act)
u_yn       = sn.Functional('u_yn', [x,y], layers, act)
sig_x_O   = sn.Functional('sigma_x',  [x, y], layers, act)
sig_y_O   = sn.Functional('sigma_y',  [x, y], layers, act)
sig_xy_O  = sn.Functional('sigma_xy', [x, y], layers, act)

# for hard boundary Mechanics
u_x = u_xn*x*(x-1.0)
u_y = u_yn*y*(y-1.0)

strain_x  = diff(u_x, x)
strain_y  = diff(u_y, y)
strain_xy = (diff(u_x, y) + diff(u_y, x))*0.5
alpha = 1
T_0 = 0
strain_T_x = (T-T_0)*alpha
strain_T_y = (T-T_0)*alpha
sigma_x   = (E/((1+nu)*(1-2*nu)))*((1-nu)*strain_x + nu*strain_y) - E*strain_T_x/(1-2*nu)
sigma_y   = (E/((1+nu)*(1-2*nu)))*((1-nu)*strain_y + nu*strain_x) - E*strain_T_y/(1-2*nu)
sigma_xy  = (E/((1+nu)*(1-2*nu)))*strain_xy*(1-2*nu)

"""**Inputs generation**"""

# NUM_SAMPLES = 10000        # 100000
# dg = DataGeneratorXY(
#     X=[0, 1], Y=[0, 1],
#     num_sample=NUM_SAMPLES,
#     targets=['all','all', 'all', 'all', 'all','all',
#             'bc-left','bc-bot', 'bc-right', 'bc-top',
#              'bc-bot', 'bc-bot','bc-bot','bc-bot', 'bc-top', 'bc-top','bc-top','bc-top',
#              'all','all', 'all', 'all', 'bc-left', 'bc-right',
#              'bc-bot','bc-bot', 'bc-top', 'bc-top']
# )
# input_data_dg, _ = dg.get_data()

Load_data = np.loadtxt('node.txt')

x1 = Load_data[:,2]
y1 = Load_data[:,3]
e1 = Load_data[:,4]

x_data = np.reshape(Load_data[:,2], (51,51))
y_data = np.reshape(Load_data[:,3], (51,51))

E_data = np.reshape(e1, (51,51))
K_data = E_data


### adding collocation point to BC Top
for iii in range(0,400):
    x_new = np.random.random()
    y_new = 1
    e_new = 1
    x1 = np.append(x1, x_new)
    y1 = np.append(y1, y_new)
    e1 = np.append(e1, e_new)
### adding collocation point to BC bottom
for iiii in range(0,400):
    x_new = np.random.random()
    y_new = 0
    e_new = 1
    x1 = np.append(x1, x_new)
    y1 = np.append(y1, y_new)
    e1 = np.append(e1, e_new)
### adding collocation point to BC right
for iiii in range(0,400):
    x_new = 1
    y_new = np.random.random()
    e_new = 1
    x1 = np.append(x1, x_new)
    y1 = np.append(y1, y_new)
    e1 = np.append(e1, e_new)
### adding collocation point to BC left
for iiii in range(0,400):
    x_new = 0
    y_new = np.random.random()
    e_new = 1
    x1 = np.append(x1, x_new)
    y1 = np.append(y1, y_new)
    e1 = np.append(e1, e_new)
k1 = e1

# E_data = np.full_like(input_data_dg[0],0.3)
# K_data = np.full_like(input_data_dg[0],0.3)

# for i in range(len(input_data_dg[0])):
#     if input_data_dg[0][i]>0.25 and input_data_dg[0][i]<0.5 and input_data_dg[1][i]>0.625 and input_data_dg[1][i]<0.875:
#         E_data[i] = 1.0
#         K_data[i] = 1.0

# fig= plt.figure(figsize=(5,5))
# plt.scatter(input_data_dg[0],input_data_dg[1],E_data, E_data)

w1 = 1
w2 = 1
w3 = 1
w4 = 1
w5 = 1
wN = 1

def boundary_value_mask(x,y):
    M = np.zeros(x.shape)
    for i in range(len(x)):
        if x[i]==0 or x[i]==1 or y[i]==0 or y[i]==1:
            M[i] = 1
    return M

def boundary_value_mask_right(x,y):
    M = np.zeros(x.shape)
    for i in range(len(x)):
        if x[i]==1:
            M[i] = 1
    return M

def boundary_value_mask_left(x,y):
    M = np.zeros(x.shape)
    for i in range(len(x)):
        if x[i]==0:
            M[i] = 1
    return M

def boundary_value_mask_top(x,y):
    M = np.zeros(x.shape)
    for i in range(len(x)):
        if y[i]==1:
            M[i] = 1
    return M

def boundary_value_mask_bottom(x,y):
    M = np.zeros(x.shape)
    for i in range(len(x)):
        if y[i]==0:
            M[i] = 1
    return M

M_bound = boundary_value_mask(x1,y1)
M_bound_right = boundary_value_mask_right(x1,y1)
M_bound_left = boundary_value_mask_left(x1,y1)
M_bound_top = boundary_value_mask_top(x1,y1)
M_bound_bottom = boundary_value_mask_bottom(x1,y1)
M_domain = np.array((M_bound==0)*1.0)

nb = np.sum(M_bound)
nb_r = np.sum(M_bound_right)
nb_l = np.sum(M_bound_left)
nb_t = np.sum(M_bound_top)
nb_b = np.sum(M_bound_bottom)
nd = np.sum(M_domain)

M_bound_right = tf.convert_to_tensor(M_bound_right)
M_bound_left = tf.convert_to_tensor(M_bound_left)
M_bound_top = tf.convert_to_tensor(M_bound_top)
M_bound_bottom = tf.convert_to_tensor(M_bound_bottom)
M_domain = tf.convert_to_tensor(M_domain)

# Mechanical Govenring equation
################# ENERGY 
Work_int_m = np.sum(M_domain*(sigma_x * strain_x + sigma_y * strain_y + sigma_xy * 2* strain_xy), axis=-1)/(2*nd)

Work_e_r_m = np.sum(M_bound_right*(sigma_x * u_x + sigma_xy * u_y), axis=-1)/nb_r
Work_e_l_m = np.sum(M_bound_left*(-sigma_x * u_x - sigma_xy * u_y), axis=-1)/nb_l
Work_e_t_m = np.sum(M_bound_top*(sigma_y * u_y + sigma_xy * u_x), axis=-1)/nb_t
Work_e_b_m = np.sum(M_bound_bottom*(-sigma_y * u_y - sigma_xy * u_x), axis=-1)/nb_b
Work_ext_m = Work_e_r_m + Work_e_l_m + Work_e_t_m + Work_e_b_m

Work_m = sn.PDE(sqrt(abs(w1*Work_int_m - w1*Work_ext_m)))

################# PDE
PDE1_m = sn.PDE(w2*diff(sig_x_O, x) + w2*diff(sig_xy_O, y))
PDE2_m = sn.PDE(w2*diff(sig_xy_O, x) + w2*diff(sig_y_O, y))

################# CONECT
EQ1_m  = sn.PDE(w3*sig_x_O  - w3*sigma_x)
EQ2_m  = sn.PDE(w3*sig_y_O  - w3*sigma_y)
EQ3_m  = sn.PDE(w3*sig_xy_O - w3*sigma_xy)

################# Mechanical BCs 
BC1_m  = (x==0.)*(w4*u_x)         #left disp_x
BC2_m  = (y==0.)*(w4*u_y)         #bot disp_y

BC3_m  = (x==1.)*(w4*u_x)         #right disp_x
BC4_m  = (y==1.)*(w4*u_y)         #top disp_y

# BC5_m  = (x==0.)*(w5*sigma_y)     #down sig_u
BC6_m  = (x==0.)*(w5*sigma_xy)    #down sig_u
# BC7_m  = (x==0.)*(wN*sig_y_O)     #down sig_O
BC8_m  = (x==0.)*(wN*sig_xy_O)   #down sig_O

# BC9_m  = (x==1.)*(w5*sigma_y)     #up sig_u
BC10_m = (x==1.)*(w5*sigma_xy)    #up sig_u
# BC11_m = (x==1.)*(wN*sig_y_O)     #up sig_O
BC12_m = (x==1.)*(wN*sig_xy_O)    #up sig_O

# BC13_m  = (y==0.)*(w5*sigma_x)     #down sig_u
BC14_m  = (y==0.)*(w5*sigma_xy)    #down sig_u
# BC15_m  = (y==0.)*(wN*sig_x_O)     #down sig_O
BC16_m  = (y==0.)*(wN*sig_xy_O)   #down sig_O

# BC17_m  = (y==1.)*(w5*sigma_x)     #up sig_u
BC18_m  = (y==1.)*(w5*sigma_xy)    #up sig_u
# BC19_m  = (y==1.)*(wN*sig_x_O)     #up sig_O
BC20_m  = (y==1.)*(wN*sig_xy_O)    #up sig_O

# Thermal Governing Equation
Work_int_t = np.sum(M_domain*(q_x * gr_T_x + q_y * gr_T_y), axis=-1)/(2*nd)

Work_e_r_t = np.sum(M_bound_right*(q_x * T), axis=-1)/nb_r
Work_e_l_t = np.sum(M_bound_left*(-q_x * T), axis=-1)/nb_l
Work_e_t_t = np.sum(M_bound_top*(q_y * T), axis=-1)/nb_t
Work_e_b_t = np.sum(M_bound_bottom*(-q_y * T), axis=-1)/nb_b
Work_ext_t = Work_e_r_t + Work_e_l_t + Work_e_t_t + Work_e_b_t

Work_t = sn.PDE(sqrt(abs(w1*Work_int_t - w1*Work_ext_t)))


################# PDE
PDE1_t = sn.PDE(w2*diff(q_x_O, x) + w2*diff(q_y_O, y))


################# CONECT
EQ1_t  = sn.PDE(w3*q_x_O  - w3*q_x)
EQ2_t  = sn.PDE(w3*q_y_O  - w3*q_y)

################# Thermal BCs
BC1_t  = (x==0.)*(w4*T - 1)                #left
BC2_t  = (x==1.)*(w4*T)                #right temp

# BC3_t  = (y==0.)*(w4*T)                #up
# BC4_t  = (y==1.)*(w4*T)                #down temp

BC5_t  = (y==0.)*(w5*q_y)              #down q_y
BC6_t  = (y==0.)*(wN*q_y_O)            #down q_y

BC7_t = (y==1.)*(w5*q_y)               #up q_y
BC8_t = (y==1.)*(wN*q_y_O)             #up q_y

model = sn.SciModel([x, y, E, K], [Work_m, PDE1_m, PDE2_m, EQ1_m, EQ2_m, EQ3_m,
                                BC6_m, BC8_m, BC10_m,
                                BC12_m, BC14_m, BC16_m, BC18_m, BC20_m,
                                Work_t, PDE1_t, EQ1_t, EQ2_t, BC5_t, BC6_t, BC7_t, BC8_t], loss_func="mse",optimizer=Opt)

# print(model.summary())

input_data = [x1, y1, e1, k1]
target_data = 22*['zeros']

history = model.train(input_data, target_data, learning_rate=0.001, epochs=10000, batch_size=5000, verbose=2)

model.save_weights('model_v_new.hdf5')
# model.load_weights('model_v_new.hdf5')

loss_name = ['Total_loss', 'Work_m', 'PDE1_m', 'PDE2_m', 'EQ1_m', 'EQ2_m', 'EQ3_m',
                                'BC6_m',
                                'BC8_m', 'BC10_m', 'BC12_m','BC14_m',
                                'BC16_m', 'BC18_m', 'BC20_m', 
                                'Work_t', 'PDE1_t', 'EQ1_t', 'EQ2_t', 'BC1_t',
                                'BC2_t', 'BC5_t', 'BC6_t', 'BC7_t', 'BC8_t']
fig = plt.figure(figsize=(12, 7))
itter = 0 
for word, loss in history.history.items():
        if word.endswith("loss"):
            plt.semilogy(np.array(loss), label=loss_name[itter])
            itter+=1
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('losses')

fig = plt.figure(figsize=(12, 7))
plt.semilogy(history.history['loss'], label='Total_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Total-loss')
plt.savefig('totalloss')

result_FE = pd.read_csv('rectan_inclus.csv')

# x_x =  pd.DataFrame(result_FE, columns= ['x'])
# y_y =  pd.DataFrame(result_FE, columns= ['y'])
u_x_FE =  pd.DataFrame(result_FE, columns= ['Displacements:0'])
u_y_FE =  pd.DataFrame(result_FE, columns= ['Displacements:1'])
sigma_x_FE =  pd.DataFrame(result_FE, columns= ['Stress/Strain:0'])
sigma_y_FE =  pd.DataFrame(result_FE, columns= ['Stress/Strain:1'])
sigma_xy_FE =  pd.DataFrame(result_FE, columns= ['Stress/Strain:3'])
T_FE =  pd.DataFrame(result_FE, columns= ['Displacements:2'])
q_x_FE =  pd.DataFrame(result_FE, columns= ['Stress/Strain:12'])
q_y_FE =  pd.DataFrame(result_FE, columns= ['Stress/Strain:13'])

u_x_FE = np.array(u_x_FE)
u_x_FE = np.reshape(u_x_FE, (51,51))
u_y_FE = np.array(u_y_FE)
u_y_FE = np.reshape(u_y_FE, (51,51))
sigma_x_FE = np.array(sigma_x_FE)
sigma_x_FE = np.reshape(sigma_x_FE, (51,51))
sigma_y_FE = np.array(sigma_y_FE)
sigma_y_FE = np.reshape(sigma_y_FE, (51,51))
sigma_xy_FE = np.array(sigma_xy_FE)
sigma_xy_FE = np.reshape(sigma_xy_FE, (51,51))

T_FE = np.array(T_FE)
T_FE = np.reshape(T_FE, (51,51))
q_x_FE = np.array(q_x_FE)
q_x_FE = np.reshape(q_x_FE, (51,51))
q_y_FE = np.array(q_y_FE)
q_y_FE = np.reshape(q_y_FE, (51,51))

u_x_pred = u_x.eval(model, [x_data,y_data,E_data,K_data])
u_y_pred = u_y.eval(model, [x_data,y_data,E_data,K_data])

sigma_x_pred  = sigma_x.eval(model, [x_data,y_data,E_data,K_data])
sigma_y_pred  = sigma_y.eval(model, [x_data,y_data,E_data,K_data])
sigma_xy_pred = sigma_xy.eval(model, [x_data,y_data,E_data,K_data])

sig_x_O_pred  = sig_x_O.eval(model, [x_data,y_data,E_data,K_data])
sig_y_O_pred  = sig_y_O.eval(model, [x_data,y_data,E_data,K_data])
sig_xy_O_pred = sig_xy_O.eval(model, [x_data,y_data,E_data,K_data])

strain_x_pred = strain_x.eval(model, [x_data,y_data,E_data,K_data])
strain_y_pred = strain_y.eval(model, [x_data,y_data,E_data,K_data])
strain_xy_pred = strain_xy.eval(model, [x_data,y_data,E_data,K_data])

u_x_error = np.abs(u_x_FE - u_x_pred)
u_y_error = np.abs(u_y_FE - u_y_pred)
sigma_x_error   = np.abs(sigma_x_FE - sigma_x_pred)
sigma_y_error   = np.abs(sigma_y_FE - sigma_y_pred)
sigma_xy_error  = np.abs(sigma_xy_FE - sigma_xy_pred)
# strain_x_error  = np.abs(strain_x_FE - strain_x_pred)
# strain_y_error  = np.abs(strain_y_FE - strain_y_pred)
# strain_xy_error = np.abs(strain_xy_FE - strain_xy_pred)

T_pred     = T.eval(model, [x_data,y_data,E_data,K_data])
q_x_pred   = q_x.eval(model, [x_data,y_data,E_data,K_data])
q_y_pred   = q_y.eval(model, [x_data,y_data,E_data,K_data])
q_x_O_pred = q_x_O.eval(model, [x_data,y_data,E_data,K_data])
q_y_O_pred = q_y_O.eval(model, [x_data,y_data,E_data,K_data])
##### computing the errors 
T_error = np.abs(T_pred- T_FE)
q_x_error = np.abs(q_x_pred- q_x_FE)
q_y_error = np.abs(q_y_pred- q_y_FE)

fig,ax = plt.subplots(2,3,figsize=(12,8))
plt.colorbar(ax[0,0].pcolor(x_data, y_data, u_x_pred, cmap='jet'),ax=ax[0,0])
plt.colorbar(ax[0,1].pcolor(x_data, y_data, u_x_FE, cmap='jet'),ax=ax[0,1])
plt.colorbar(ax[0,2].pcolor(x_data, y_data, u_x_error, cmap='jet'),ax=ax[0,2])
plt.colorbar(ax[1,0].pcolor(x_data, y_data, u_y_pred, cmap='jet'),ax=ax[1,0])
plt.colorbar(ax[1,1].pcolor(x_data, y_data, u_y_FE, cmap='jet'),ax=ax[1,1])
plt.colorbar(ax[1,2].pcolor(x_data, y_data, u_y_error, cmap='jet'),ax=ax[1,2])
ax[0,0].set_title('u_x_pred')
ax[0,1].set_title('u_x_FE')
ax[0,2].set_title('u_x_error')
ax[1,0].set_title('u_y_pred')
ax[1,1].set_title('u_y_FE')
ax[1,2].set_title('u_y_error')
ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[1,1].set_yticks([])
ax[1,2].set_yticks([])
# print("Mean Abs Error of u_x = " , np.mean(u_x_error))
# print("Mean Abs Error of u_y = " , np.mean(u_y_error))
plt.savefig('disp')

fig,ax = plt.subplots(3,4,figsize=(16,12))
plt.colorbar(ax[0,0].pcolor(x_data, y_data, sigma_x_pred, cmap='jet'),ax=ax[0,0])
plt.colorbar(ax[0,1].pcolor(x_data, y_data, sig_x_O_pred, cmap='jet'),ax=ax[0,1])
plt.colorbar(ax[0,2].pcolor(x_data, y_data, sigma_x_FE, cmap='jet'),ax=ax[0,2])
plt.colorbar(ax[0,3].pcolor(x_data, y_data, sigma_x_error, cmap='jet'),ax=ax[0,3])
plt.colorbar(ax[1,0].pcolor(x_data, y_data, sigma_y_pred, cmap='jet'),ax=ax[1,0])
plt.colorbar(ax[1,1].pcolor(x_data, y_data, sig_y_O_pred, cmap='jet'),ax=ax[1,1])
plt.colorbar(ax[1,2].pcolor(x_data, y_data, sigma_y_FE, cmap='jet'),ax=ax[1,2])
plt.colorbar(ax[1,3].pcolor(x_data, y_data, sigma_y_error, cmap='jet'),ax=ax[1,3])
plt.colorbar(ax[2,0].pcolor(x_data, y_data, sigma_xy_pred, cmap='jet'),ax=ax[2,0])
plt.colorbar(ax[2,1].pcolor(x_data, y_data, sig_xy_O_pred, cmap='jet'),ax=ax[2,1])
plt.colorbar(ax[2,2].pcolor(x_data, y_data, sigma_xy_FE, cmap='jet'),ax=ax[2,2])
plt.colorbar(ax[2,3].pcolor(x_data, y_data, sigma_xy_error, cmap='jet'),ax=ax[2,3])
ax[0,0].set_title('sigma_x_pred')
ax[0,1].set_title('sig_x_O_pred')
ax[0,2].set_title('sigma_x_FE')
ax[0,3].set_title('sigma_x_error')
ax[1,0].set_title('sigma_y_pred')
ax[1,1].set_title('sig_y_O_pred')
ax[1,2].set_title('sigma_y_FE')
ax[1,3].set_title('sigma_y_error')
ax[2,0].set_title('sigma_xy_pred')
ax[2,1].set_title('sig_xy_O_pred')
ax[2,2].set_title('sigma_xy_FE')
ax[2,3].set_title('sigma_xy_error')
ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[0,3].set_yticks([])
ax[1,2].set_yticks([])
ax[1,1].set_yticks([])
ax[1,3].set_yticks([])
ax[2,1].set_yticks([])
ax[2,2].set_yticks([])
ax[2,3].set_yticks([])
# print("Mean Abs Error of sigma_x = " , np.mean(sigma_x_error))
# print("Mean Abs Error of sigma_y = " , np.mean(sigma_y_error))
# print("Mean Abs Error of sigma_xy = " , np.mean(sigma_xy_error))
plt.savefig('stress')

fac = 1
x_new_pred = x_data + u_x_pred*fac
y_new_pred = y_data + u_y_pred*fac
x_new_FE = x_data + u_x_FE*fac
y_new_FE = y_data + u_y_FE*fac

fig,ax = plt.subplots(1,3,figsize=(20,7))
ax[0].plot(x_data, y_data,'b.')
ax[1].plot(x_new_pred, y_new_pred, 'r.')
ax[2].plot(x_new_FE, y_new_FE, 'r.')
ax[0].set_title('Undeformed')
ax[1].set_title('Deformed_pred')
ax[2].set_title('Deformed_FE')
plt.savefig('deform')

fig,ax = plt.subplots(1,3,figsize=(15,4))
plt.colorbar(ax[0].pcolor(x_data, y_data, T_pred, cmap='jet'),ax=ax[0])
plt.colorbar(ax[1].pcolor(x_data, y_data, T_FE, cmap='jet'),ax=ax[1])
plt.colorbar(ax[2].pcolor(x_data, y_data, T_error, cmap='jet'),ax=ax[2])
ax[0].set_title('T_Pred')
ax[1].set_title('T_FE')
ax[2].set_title('T_Error')
plt.savefig('temp')

fig,ax = plt.subplots(2,4,figsize=(18,8))
plt.colorbar(ax[0,0].pcolor(x_data, y_data, q_x_pred, cmap='jet'),ax=ax[0,0])
plt.colorbar(ax[0,1].pcolor(x_data, y_data, q_x_O_pred, cmap='jet'),ax=ax[0,1])
plt.colorbar(ax[0,2].pcolor(x_data, y_data, q_x_FE, cmap='jet'),ax=ax[0,2])
plt.colorbar(ax[0,3].pcolor(x_data, y_data, q_x_error, cmap='jet'),ax=ax[0,3])

plt.colorbar(ax[1,0].pcolor(x_data, y_data, q_y_pred, cmap='jet'),ax=ax[1,0])
plt.colorbar(ax[1,1].pcolor(x_data, y_data, q_y_O_pred, cmap='jet'),ax=ax[1,1])
plt.colorbar(ax[1,2].pcolor(x_data, y_data, q_y_FE, cmap='jet'),ax=ax[1,2])
plt.colorbar(ax[1,3].pcolor(x_data, y_data, q_y_error, cmap='jet'),ax=ax[1,3])

ax[0,0].set_title('q_x_pred')
ax[0,1].set_title('q_x_O_pred')
ax[0,2].set_title('q_x_FE')
ax[1,0].set_title('q_y_pred')
ax[1,1].set_title('q_y_O_pred')
ax[1,1].set_title('q_y_FE')
ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[1,1].set_yticks([])
ax[1,2].set_yticks([])
plt.savefig('Fluxes')

all_losses = pd.DataFrame(history.history)
itter = 0 
for word, loss in history.history.items():
    if word.endswith("loss"):
        all_losses = all_losses.rename(columns={word:loss_name[itter]})
        itter+=1
all_losses.to_csv('all_losses.csv')

fig,ax = plt.subplots(5,3,figsize=(12,18))
cb= plt.colorbar(ax[0,0].pcolor(x_data, y_data, u_x_pred, cmap='rainbow', vmin=u_x_FE.min(), vmax=u_x_FE.max(), linewidth=0,rasterized=True),ax=ax[0,0], ticks=[u_x_FE.min(),u_x_FE.max()])
cb1=plt.colorbar(ax[0,1].pcolor(x_data, y_data, u_x_FE, cmap='rainbow', vmin=u_x_FE.min(), vmax=u_x_FE.max(), linewidth=0,rasterized=True),ax=ax[0,1], ticks=[u_x_FE.min(),u_x_FE.max()])
cb2=plt.colorbar(ax[0,2].pcolor(x_data, y_data, u_x_error, cmap='rainbow',vmin=0, vmax=np.round(u_x_error.max(),4), linewidth=0,rasterized=True),ax=ax[0,2], ticks=[0,np.round(u_x_error.max(),4)])

cb3=plt.colorbar(ax[1,0].pcolor(x_data, y_data, u_y_pred, cmap='rainbow', vmin=np.round(u_y_FE.min(),4), vmax=np.round(u_y_FE.max(),4), linewidth=0,rasterized=True),ax=ax[1,0], ticks=[np.round(u_y_FE.min(),4),np.round(u_y_FE.max(),4)])
cb4=plt.colorbar(ax[1,1].pcolor(x_data, y_data, u_y_FE, cmap='rainbow', vmin=np.round(u_y_FE.min(),4), vmax=np.round(u_y_FE.max(),4), linewidth=0,rasterized=True),ax=ax[1,1], ticks=[np.round(u_y_FE.min(),4),np.round(u_y_FE.max(),4)])
cb5=plt.colorbar(ax[1,2].pcolor(x_data, y_data, u_y_error, cmap='rainbow',vmin=0, vmax=np.round(u_y_error.max(),4), linewidth=0,rasterized=True),ax=ax[1,2], ticks=[0,np.round(u_y_error.max(),4)])

cb6=plt.colorbar(ax[2,0].pcolor(x_data, y_data, sigma_x_pred, cmap='jet', vmin=np.round(sigma_x_FE.min(),4), vmax=np.round(sigma_x_FE.max(),4), linewidth=0,rasterized=True),ax=ax[2,0], ticks=[np.round(sigma_x_FE.min(),4),np.round(sigma_x_FE.max(),4)])
cb7=plt.colorbar(ax[2,1].pcolor(x_data, y_data, sigma_x_FE, cmap='jet', vmin=np.round(sigma_x_FE.min(),4), vmax=np.round(sigma_x_FE.max(),4), linewidth=0,rasterized=True),ax=ax[2,1], ticks=[np.round(sigma_x_FE.min(),4),np.round(sigma_x_FE.max(),4)])
cb8=plt.colorbar(ax[2,2].pcolor(x_data, y_data, sigma_x_error, cmap='jet', vmin=np.round(sigma_x_error.min(),4), vmax=np.round(sigma_x_error.max(),4), linewidth=0,rasterized=True),ax=ax[2,2], ticks=[np.round(sigma_x_error.min(),4),np.round(sigma_x_error.max(),4)])


cb9=plt.colorbar(ax[3,0].pcolor(x_data, y_data, sigma_y_pred, cmap='jet', vmin=np.round(sigma_y_FE.min(),4), vmax=np.round(sigma_y_FE.max(),4), linewidth=0,rasterized=True),ax=ax[3,0], ticks=[np.round(sigma_y_FE.min(),4),np.round(sigma_y_FE.max(),4)])
cb10=plt.colorbar(ax[3,1].pcolor(x_data, y_data, sigma_y_FE, cmap='jet',vmin=sigma_y_FE.min(), vmax=np.round(sigma_y_FE.max(),4),linewidth=0,rasterized=True),ax=ax[3,1], ticks=[np.round(sigma_y_FE.min(),4),np.round(sigma_y_FE.max(),4)])
cb11=plt.colorbar(ax[3,2].pcolor(x_data, y_data, sigma_y_error, cmap='jet', vmin=np.round(sigma_y_error.min(),4), vmax=np.round(sigma_y_error.max(),4), linewidth=0,rasterized=True),ax=ax[3,2], ticks=[np.round(sigma_y_error.min(),4),np.round(sigma_y_error.max(),4)])

cb12=plt.colorbar(ax[4,0].pcolor(x_data, y_data, sigma_xy_pred, cmap='jet', vmin=np.round(sigma_xy_pred.min(),4), vmax=np.round(sigma_xy_FE.max(),4), linewidth=0,rasterized=True),ax=ax[4,0], ticks=[np.round(sigma_xy_pred.min(),4),np.round(sigma_xy_FE.max(),4)])
cb13=plt.colorbar(ax[4,1].pcolor(x_data, y_data, sigma_xy_FE, cmap='jet', vmin=np.round(sigma_xy_FE.min(),4),vmax=np.round(sigma_xy_FE.max(),4),linewidth=0,rasterized=True),ax=ax[4,1], ticks=[np.round(sigma_xy_FE.min(),4),np.round(sigma_xy_FE.max(),4)])
cb14=plt.colorbar(ax[4,2].pcolor(x_data, y_data, sigma_xy_error, cmap='jet', vmin=0,vmax=sigma_xy_error.max(), linewidth=0,rasterized=True),ax=ax[4,2], ticks=[0,sigma_xy_error.max()])
ax[0,0].set_title(r'$u_x \, \, \, \, PINNs$', fontsize=14), ax[0,1].set_title(r'$u_x \, \, \, \, FE$', fontsize=14), ax[0,2].set_title(r'$u_x \, \, \, \, error$', fontsize=14)
ax[1,0].set_title(r'$u_y \, \, \, \, PINNs$', fontsize=14), ax[1,1].set_title(r'$u_y \, \, \, \, FE$', fontsize=14), ax[1,2].set_title(r'$u_y \, \, \, \, error$', fontsize=14)
ax[2,0].set_title(r"$\sigma_x \, \, \, \, PINNs$ ", fontsize=14), ax[2,1].set_title(r"$\sigma_x \, \, \, \, FE$ ", fontsize=14), ax[2,2].set_title(r"$\sigma_x \, \, \, \, error$ ", fontsize=14)
ax[3,0].set_title(r"$\sigma_y \, \, \, \, PINNs$ ", fontsize=14), ax[3,1].set_title(r"$\sigma_y \, \, \, \, FE$ ", fontsize=14), ax[3,2].set_title(r"$\sigma_y \, \, \, \, error$ ", fontsize=14)
ax[4,0].set_title(r"$\sigma_{xy} \, \, \, \, PINNs$ ", fontsize=14), ax[4,1].set_title(r"$\sigma_{xy} \, \, \, \, FE$ ", fontsize=14), ax[4,2].set_title(r"$\sigma_{xy} \, \, \, \, error$ ", fontsize=14)
ax[0,0].set_yticks([]), ax[0,1].set_yticks([]), ax[0,2].set_yticks([])
ax[1,0].set_yticks([]), ax[1,1].set_yticks([]), ax[1,2].set_yticks([])
ax[2,0].set_yticks([]), ax[2,1].set_yticks([]), ax[2,2].set_yticks([])
ax[3,0].set_yticks([]), ax[3,1].set_yticks([]), ax[3,2].set_yticks([])
ax[4,0].set_yticks([]), ax[4,1].set_yticks([]), ax[4,2].set_yticks([])
ax[0,0].set_xticks([]), ax[0,1].set_xticks([]), ax[0,2].set_xticks([])
ax[1,0].set_xticks([]), ax[1,1].set_xticks([]), ax[1,2].set_xticks([])
ax[2,0].set_xticks([]), ax[2,1].set_xticks([]), ax[2,2].set_xticks([])
ax[3,0].set_xticks([]), ax[3,1].set_xticks([]), ax[3,2].set_xticks([])
ax[4,0].set_xticks([]), ax[4,1].set_xticks([]), ax[4,2].set_xticks([])
cb.ax.tick_params(labelsize=12),cb1.ax.tick_params(labelsize=12),cb2.ax.tick_params(labelsize=12)
cb3.ax.tick_params(labelsize=12),cb4.ax.tick_params(labelsize=12),cb5.ax.tick_params(labelsize=12)
cb6.ax.tick_params(labelsize=12),cb7.ax.tick_params(labelsize=12),cb8.ax.tick_params(labelsize=12)
cb9.ax.tick_params(labelsize=12),cb10.ax.tick_params(labelsize=12),cb11.ax.tick_params(labelsize=12)
cb12.ax.tick_params(labelsize=12),cb13.ax.tick_params(labelsize=12),cb14.ax.tick_params(labelsize=12)
plt.savefig("mechanic_couple.pdf")
plt.show()

fig,ax = plt.subplots(3,3,figsize=(12,10.5))
plt.colorbar(ax[0,0].pcolor(x_data, y_data, T_pred, cmap='rainbow',vmin=T_FE.min(), vmax=T_FE.max(), linewidth=0,rasterized=True)
                            ,ax=ax[0,0], ticks=[T_FE.min(),T_FE.max()])
plt.colorbar(ax[0,1].pcolor(x_data, y_data, T_FE, cmap='rainbow',vmin=T_FE.min(), vmax=T_FE.max(), linewidth=0,rasterized=True) 
                            ,ax=ax[0,1], ticks=[T_FE.min(),T_FE.max()])
plt.colorbar(ax[0,2].pcolor(x_data, y_data, T_error, cmap='rainbow',vmin=T_error.min(), vmax=T_error.max(), linewidth=0,rasterized=True)
                            ,ax=ax[0,2], ticks=[T_error.min(),T_error.max()])

plt.colorbar(ax[1,0].pcolor(x_data, y_data, q_x_pred, cmap='jet', vmin=q_x_FE.min(), vmax=q_x_FE.max(), linewidth=0,rasterized=True)
                            ,ax=ax[1,0], ticks=[q_x_FE.min(),q_x_FE.max()])
plt.colorbar(ax[1,1].pcolor(x_data, y_data, q_x_FE, cmap='jet',vmin=q_x_FE.min(), vmax=q_x_FE.max(), linewidth=0,rasterized=True)
                            ,ax=ax[1,1], ticks=[q_x_FE.min(),q_x_FE.max()])
plt.colorbar(ax[1,2].pcolor(x_data, y_data, q_x_error, cmap='jet',vmin=q_x_error.min(), vmax=q_x_error.max(), linewidth=0,rasterized=True)
                            ,ax=ax[1,2], ticks=[q_x_error.min(),q_x_error.max()])

plt.colorbar(ax[2,0].pcolor(x_data, y_data, q_y_pred, cmap='jet',vmin=q_y_FE.min(), vmax=q_y_FE.max(),linewidth=0,rasterized=True)
                            ,ax=ax[2,0],ticks=[q_y_FE.min(),q_y_FE.max()])
plt.colorbar(ax[2,1].pcolor(x_data, y_data, q_y_FE, cmap='jet',vmin=q_y_FE.min(), vmax=q_y_FE.max(),linewidth=0,rasterized=True)
                            ,ax=ax[2,1],ticks=[q_y_FE.min(),q_y_FE.max()])
plt.colorbar(ax[2,2].pcolor(x_data, y_data, q_y_error, cmap='jet',vmin=q_y_error.min(), vmax=q_y_error.max(),linewidth=0,rasterized=True)
                            ,ax=ax[2,2],ticks=[q_y_error.min(),q_y_error.max()])

ax[0,0].set_title(r'$T \, \, \, \, PINNs$'),ax[0,1].set_title(r'$T \, \, \, \, FE$'),ax[0,2].set_title(r'$T \, \, \, \, error$')
ax[1,0].set_title(r'$q_x \, \, \, \, PINNs$'),ax[1,1].set_title(r'$q_x \, \, \, \, FE$'),ax[1,2].set_title(r'$q_x \, \, \, \, error$')
ax[2,0].set_title(r'$q_y \, \, \, \, PINNs$'),ax[2,1].set_title(r'$q_y \, \, \, \, FE$'),ax[2,2].set_title(r'$q_y \, \, \, \, error$')
ax[0,0].set_yticks([]), ax[0,1].set_yticks([]), ax[0,2].set_yticks([])
ax[1,0].set_yticks([]), ax[1,1].set_yticks([]), ax[1,2].set_yticks([])
ax[2,0].set_yticks([]), ax[2,1].set_yticks([]), ax[2,2].set_yticks([])
ax[0,0].set_xticks([]), ax[0,1].set_xticks([]), ax[0,2].set_xticks([])
ax[1,0].set_xticks([]), ax[1,1].set_xticks([]), ax[1,2].set_xticks([])
ax[2,0].set_xticks([]), ax[2,1].set_xticks([]), ax[2,2].set_xticks([])
plt.savefig('temperature_couple.pdf')
