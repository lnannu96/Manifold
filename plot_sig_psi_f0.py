import numpy as np 
import matplotlib.pyplot as plt 

data_k1 = np.genfromtxt("f0_k1.txt")
data_k2 = np.genfromtxt("f0_k2.txt")
data_k3 = np.genfromtxt("f0_k3.txt")
data_k4 = np.genfromtxt("f0_k4.txt")

xs = np.array([0.0,0.5,1.0,2.5,5.0,7.5,10.0])

fig,ax = plt.subplots(2,2)

fig.suptitle('Error vs. $\psi$ for Various k')

ax[0,0].plot(xs,data_k1[:,0],'k-',label='k=1')
ax[0,0].fill_between(xs,data_k1[:,0]-data_k1[:,1],data_k1[:,0]+data_k1[:,1],alpha=0.5)

ax[0,1].plot(xs,data_k2[:,0],'k-',label='k=2')
ax[0,1].fill_between(xs,data_k2[:,0]-data_k2[:,1],data_k2[:,0]+data_k2[:,1],alpha=0.5)

ax[1,0].plot(xs,data_k3[:,0],'k-',label='k=3')
ax[1,0].fill_between(xs,data_k3[:,0]-data_k3[:,1],data_k3[:,0]+data_k3[:,1],alpha=0.5)

ax[1,1].plot(xs,data_k4[:,0],'k-',label='k=4')
ax[1,1].fill_between(xs,data_k4[:,0]-data_k4[:,1],data_k4[:,0]+data_k4[:,1],alpha=0.5)

ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()

ax[0,0].set_ylabel('$\sigma \pm \delta_\sigma$')
ax[1,0].set_ylabel('$\sigma \pm \delta_\sigma$')

ax[1,0].set_xlabel('-$\psi$')
ax[1,1].set_xlabel('-$\psi$')

plt.show()
