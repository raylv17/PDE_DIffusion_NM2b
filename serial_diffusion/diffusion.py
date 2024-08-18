from ftcs import *
import time

L = 5
T = 2

h = (2*L)/223 #0.01
tau = 0.001
print(f"tau:{tau}, h:{h}")
if tau/((L/h)**2) >= 0.5:
    print("make sure: tau/(L/h**2) < 0.5, else incorrect results!")

N = int(T/tau) # time-steps
v0 = lambda x : float(abs(x)<1.5)
D = lambda x : 1
S = lambda x,t : 0

t1 = time.time()
[V1,x,t] = ftcs(L,N,h,tau,D,S,v0,name="out_DS0.txt")
print(time.time() - t1)
np.savetxt("V_full.csv", V1, delimiter=",")
# print(np.shape(V1))
# print(np.shape(x))
# print(np.shape(t))

# Plots in comparison with analytical solution
##################################
lt = int(N*(0.015)) # after 30 ms
sig = np.sqrt(2*D(0)*t[lt])
v_ex = 0.5*( erf((1.5-x)/(np.sqrt(2)*sig)) - erf((-1.5-x)/(np.sqrt(2)*sig)) );
plt.subplot(3,1,1)
plt.plot(x,v_ex,'r-')
plt.plot(x,V1[lt,:],'b--')
plt.grid()
plt.ylabel('v(x,t)')
plt.title(f't={t[lt]:.6f}')


lt = int(N*(0.5)) # after 1 second
sig = np.sqrt(2*D(0)*t[lt])
v_ex = 0.5*( erf((1.5-x)/(np.sqrt(2)*sig)) - erf((-1.5-x)/(np.sqrt(2)*sig)) );
plt.subplot(3,1,2)
plt.plot(x,v_ex,'r-')
plt.plot(x,V1[lt,:],'b--')
plt.grid()
plt.ylabel('v(x,t)')
plt.title(f't={t[lt]:.6f}')

lt = int(N) - 1 # at 2 seconds
sig = np.sqrt(2*D(0)*t[lt])
v_ex = 0.5*( erf((1.5-x)/(np.sqrt(2)*sig)) - erf((-1.5-x)/(np.sqrt(2)*sig)) );
plt.subplot(3,1,3)
plt.plot(x,v_ex,'r-')
plt.plot(x,V1[lt,:],'b--')
plt.grid()
plt.ylabel('v(x,t)')
plt.title(f't={t[lt]:.6f}')


plt.tight_layout()
plt.savefig("plot2.jpg",dpi=100)
# plt.show()

# 3D Plots
########################################
fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
X,Y = np.meshgrid(x,t)

surf = ax.plot_surface(X,Y,V1,linewidth=0,cmap=cm.coolwarm,antialiased=False)
plt.title("fig2")
plt.savefig("fig2",dpi=300)
# plt.show()

# Changes in D and S
#########################################
# D = lambda x : 1-abs(x)/10;
# S = lambda x,t : -4*np.double(abs(x)<0.1 and t>0.1 and  t<0.6);

# [V2,x,t] = ftcs(L,N,h,tau,D,S,v0,name="outpy2.txt");


# fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X,Y = np.meshgrid(x,t)

# surf = ax.plot_surface(X,Y,V2,linewidth=0,cmap=cm.coolwarm,antialiased=False)
# plt.title("fig2")
# plt.savefig("fig2",dpi=300)
# plt.show()
