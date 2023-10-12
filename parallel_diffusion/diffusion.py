from ftcs import *
import sys
L = 5
N = 2000
global_x = 256 # from 2^n (n > procs) e.g n=128
h = 2*L/global_x

tau = 0.001
v0 = lambda x : float(abs(x)<1.5);

D = lambda x : 1
S = lambda x,t : 0

if rank == 0: print("\n###")

time1 = MPI.Wtime()
[V1,x,t] = ftcs(L,N,global_x,tau,D,S,v0,name="out_DS0.txt");
time2 = MPI.Wtime()

global_V1 = comm.gather(V1, root = 0)
global_x  = comm.gather(x, root = 0)

if rank == 0:
    print(f"procs:{size}, Tau:{tau} h:{h}")
    print(f"time: {time2-time1}")
    V1 = np.concatenate(global_V1, axis=1)
    # print(f"rank : A | {V1[3,:]}")
    # np.savetxt("V_mat_par.csv", V1, delimiter=",")

    x = np.concatenate(global_x)
    print(f"size V1 : {np.shape(V1)}")
    print(f"size x  : {np.shape(x)}")
    print(f"size t  : {np.shape(t)}")
    # lt = int((N/10)*(10))-1;
    lt = 3;
    sig = np.sqrt(2*D(0)*t[lt]);
    v_ex = 0.5*( erf((1.5-x)/(np.sqrt(2)*sig)) - erf((-1.5-x)/(np.sqrt(2)*sig)) );
    plt.subplot(3,1,1)
    plt.plot(x,v_ex,'r-')
    plt.plot(x,V1[lt,:],'b--')
    plt.grid()
    plt.ylabel('v(x,t)')
    plt.title(f'size : {size} | t={t[lt]:.6f}')
    
    lt = 100 - 1
    # lt =int((N/10)*(2))-1;
    sig = np.sqrt(2*D(0)*t[lt]);
    v_ex = 0.5*( erf((1.5-x)/(np.sqrt(2)*sig)) - erf((-1.5-x)/(np.sqrt(2)*sig)) );
    plt.subplot(3,1,2)
    plt.plot(x,v_ex,'r-')
    plt.plot(x,V1[lt,:],'b--')
    plt.grid()
    plt.ylabel('v(x,t)')
    plt.title(f't={t[lt]:.6f}')

    lt = 2000 - 1
    # lt = int((N/10)*(5))-1;
    sig = np.sqrt(2*D(0)*t[lt]);
    v_ex = 0.5*( erf((1.5-x)/(np.sqrt(2)*sig)) - erf((-1.5-x)/(np.sqrt(2)*sig)) );
    plt.subplot(3,1,3)
    plt.plot(x,v_ex,'r-')
    plt.plot(x,V1[lt,:],'b--')
    plt.grid()
    plt.ylabel('v(x,t)')
    plt.title(f't={t[lt]:.6f}')


    plt.tight_layout()
    plt.savefig(f"p-{size}.jpg",dpi=500)
    # plt.show()
    
    fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X,Y = np.meshgrid(x,t)

    surf = ax.plot_surface(X,Y,V1,linewidth=0,cmap=cm.coolwarm,antialiased=False)
    plt.title(f"fig1_p_{size}")
    plt.savefig(f"fig1_p_{size}",dpi=500)
    # plt.show()
    
    
MPI.Finalize()