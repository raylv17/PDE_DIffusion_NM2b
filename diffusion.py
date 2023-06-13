from ftcs import *

L = 5
N = 2000

h = 0.1
tau = 0.001
v0 = lambda x : float(abs(x)<1.5);

D = lambda x : 1
S = lambda x,t : 0


[V1,x,t] = ftcs(L,N,h,tau,D,S,v0,name="out_DS0.txt");
np.savetxt("V_full.csv", V1, delimiter=",")
print(np.shape(V1))
print(np.shape(x))
print(np.shape(t))

# Plots in comparison with analytical solution
##################################
lt = 10-1;
sig = np.sqrt(2*D(0)*t[lt]);
v_ex = 0.5*( erf((1.5-x)/(np.sqrt(2)*sig)) - erf((-1.5-x)/(np.sqrt(2)*sig)) );
plt.subplot(3,1,1)
plt.plot(x,v_ex,'r-')
plt.plot(x,V1[lt,:],'b--')
plt.grid()
plt.ylabel('v(x,t)')
plt.title(f't={t[lt]:.6f}')


lt = 100-1;
sig = np.sqrt(2*D(0)*t[lt]);
v_ex = 0.5*( erf((1.5-x)/(np.sqrt(2)*sig)) - erf((-1.5-x)/(np.sqrt(2)*sig)) );
plt.subplot(3,1,2)
plt.plot(x,v_ex,'r-')
plt.plot(x,V1[lt,:],'b--')
plt.grid()
plt.ylabel('v(x,t)')
plt.title(f't={t[lt]:.6f}')

lt = 2000-1;
sig = np.sqrt(2*D(0)*t[lt]);
v_ex = 0.5*( erf((1.5-x)/(np.sqrt(2)*sig)) - erf((-1.5-x)/(np.sqrt(2)*sig)) );
plt.subplot(3,1,3)
plt.plot(x,v_ex,'r-')
plt.plot(x,V1[lt,:],'b--')
plt.grid()
plt.ylabel('v(x,t)')
plt.title(f't={t[lt]:.6f}')


plt.tight_layout()
plt.savefig("plot1.jpg",dpi=100)
plt.show()

# 3D Plots
########################################
fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
X,Y = np.meshgrid(x,t)

surf = ax.plot_surface(X,Y,V1,linewidth=0,cmap=cm.coolwarm,antialiased=False)
plt.title("fig1")
plt.savefig("fig1",dpi=300)
plt.show()

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