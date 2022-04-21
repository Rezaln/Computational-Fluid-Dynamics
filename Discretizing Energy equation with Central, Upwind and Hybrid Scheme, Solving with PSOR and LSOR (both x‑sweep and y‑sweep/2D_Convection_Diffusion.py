"""
This code was my university project 
Code by: Reza Lotfi Navaei

"""
#===================================
#              2D Convection Diffusion
#===================================
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#============================ function definition =============================
def VELOCITY(u,v):
    
    for i in range(m+1):
        for j in range(n+1):
            u[i,j]=u_max*( 1-((2*j*dy/h)-1)**2 )
            v[i,j]=0
    
    
def INITIALIZE(T):
    
    for i in range(m+1):
        for j in range(n+1):
            T[i,j]=0


def BC(T):

    for i in range(m+1):
        T[i,0]=Tw
        T[i,n]=Tw
        
    for j in range(n+1):
        T[0,j]=Ti
        T[m,j]=(4*T[m-1,j]-T[m-2,j])/3    
        

def NORM(T,T_old):
    
    c=0
    for i in range(1,m):
        for j in range(1,n):
            c=c+(T[i,j]-T_old[i,j])**2
    c=math.sqrt(c/((m-1)*(n-1)))
    return c


   
def UPWIND(a_w,a_e,a_s,a_n,a_p,u,v,r):
    
    for i in range(1,m):
        for j in range(1,n):
            
            Pe_x=abs(u[i,j]*dx/alpha)
            Pe_y=abs(v[i,j]*dy/alpha)
                     
            a_w[i,j]= 1 + Pe_x
            a_e[i,j]= 1 
            a_s[i,j]= (1 + Pe_y) * r
            a_n[i,j]= r
            a_p[i,j]= 2 + 2*r + Pe_x + r*Pe_y
           
            
def CENTRAL(a_w,a_e,a_s,a_n,a_p,u,v,r):
    
    for i in range(1,m):
        for j in range(1,n):
  
            Pe_x=abs(u[i,j]*dx/alpha)
            Pe_y=abs(v[i,j]*dy/alpha)

            a_w[i,j]= 2 + Pe_x
            a_e[i,j]= 2 - Pe_x
            a_s[i,j]= r * ( 2 + Pe_y)
            a_n[i,j]= r * ( 2 - Pe_y)
            a_p[i,j]= 4 + 4*r
             
def HYBRID(a_w,a_e,a_s,a_n,a_p,u,v,r):
    
    for i in range(1,m):
        for j in range(1,n):
            
            Pe_x=abs(u[i,j]*dx/alpha)
            Pe_y=abs(v[i,j]*dy/alpha)
            
            if Pe_x<=2 and Pe_y<=2:
                a_w[i,j]= 2 + Pe_x
                a_e[i,j]= 2 - Pe_x
                a_s[i,j]= r * ( 2 + Pe_y)
                a_n[i,j]= r * ( 2 - Pe_y)
                a_p[i,j]= 4 + 4*r
                
            elif Pe_x>2 and Pe_y>2:            
                a_w[i,j]= 1 + Pe_x
                a_e[i,j]= 1 
                a_s[i,j]= (1 + Pe_y) * r
                a_n[i,j]= r
                a_p[i,j]= 2 + 2*r + Pe_x + r*Pe_y
            elif Pe_x<=2 and Pe_y>2:
                a_w[i,j]= 2 + Pe_x
                a_e[i,j]= 2 - Pe_x          
                a_s[i,j]= (1 + Pe_y) * r
                a_n[i,j]= r
                a_p[i,j]= 4 + r*(2 + Pe_y)
                
            elif Pe_x>2 and Pe_y<=2:
                a_w[i,j]= 1 + Pe_x
                a_e[i,j]= 1        
                a_s[i,j]= r * ( 2 + Pe_y)
                a_n[i,j]= r * ( 2 - Pe_y)
                a_p[i,j]= 2 + 4*r + Pe_x
            
def PSOR(T,a_w,a_e,a_s,a_n):
    
    for i in range(1,m):
        for j in range(1,n):    
            T[i,j]=(1-w)*T[i,j]  +(w/a_p[i,j])*( a_w[i,j]*T[i-1,j]+a_e[i,j]*T[i+1,j]+a_s[i,j]*T[i,j-1]+a_n[i,j]*T[i,j+1] )
            
            
def LSOR_y(T,a_w,a_e,a_s,a_n):
    
    aa=np.zeros(m-1);bb=np.zeros(m-1);cc=np.zeros(m-1);rr=np.zeros(m-1);ss=np.zeros(m-1);
    for j in range(1,n):
        for i in range(1,m): 
            aa[i-1]=w*a_w[i,j]
            bb[i-1]=-a_p[i,j]
            cc[i-1]=w*a_e[i,j]
            rr[i-1]=-(1-w)*a_p[i,j]*T[i,j]  -w*a_s[i,j]*T[i,j-1]-w*a_n[i,j]*T[i,j+1]
            
        # implementing boundary conditions
        i=1
        rr[i-1]=rr[i-1]-aa[i-1]*T[i-1,j]
        i=m-1
        bb[i-1]=bb[i-1]+(4/3)*cc[i-1]
        aa[i-1]=aa[i-1]-(1/3)*cc[i-1]
        
        TDMA(aa,bb,cc,rr,ss)
        
        for i in range(1,m):
            T[i,j]=ss[i-1]
         
            
def LSOR_x(T,a_w,a_e,a_s,a_n):
    
    a=np.zeros(n-1);b=np.zeros(n-1);c=np.zeros(n-1);r=np.zeros(n-1);s=np.zeros(n-1);
    for i in range(1,m):
        for j in range(1,n): 
            a[j-1]=w*a_s[i,j]
            b[j-1]=-a_p[i,j]
            c[j-1]=w*a_n[i,j]
            r[j-1]=-(1-w)*a_p[i,j]*T[i,j]  -w*a_w[i,j]*T[i-1,j]-w*a_e[i,j]*T[i+1,j]
            
        # implementing boundary conditions
        j=1
        r[j-1]=r[j-1]-a[j-1]*T[i,j-1]
        j=n-1
        r[j-1]=r[j-1]-c[j-1]*T[i,j+1]
        
        TDMA(a,b,c,r,s)
        
        for j in range(1,n):
            T[i,j]=s[j-1]
       
    
def TDMA(a,b,c,r,s):

    n=len(a)
    # forward substitution
    c[0]=c[0]/b[0]
    r[0]=r[0]/b[0]
    for i in range(1,n):
        c[i]=c[i]/(b[i]-a[i]*c[i-1])
        r[i]=(r[i]-a[i]*r[i-1])/(b[i]-a[i]*c[i-1])
    
    # backward substitution
    s[n-1]=r[n-1]
    for i in reversed(range(n-1)):
        s[i]=r[i]-s[i+1]*c[i]
    

#================================ main program ================================
m=160                          # number of horizontal divisions     
n=80                          # number of vertical divisions

# matrix definition
T=np.zeros((m+1,n+1));T_old=np.zeros((m+1,n+1))
u=np.zeros((m+1,n+1));v=np.zeros((m+1,n+1))
a_w=np.zeros((m+1,n+1));a_e=np.zeros((m+1,n+1));
a_s=np.zeros((m+1,n+1));a_n=np.zeros((m+1,n+1));a_p=np.zeros((m+1,n+1))
e=np.zeros(10000)

l=6.                          # length of the channel
h=1.                          # height of the channel
dx=l/m
dy=h/n
alpha=0.1                     # diffusivity
u_max=1.5                     # maximum velocity
Tw=100                        # wall temperature
Ti=0                          # inlet temperature
w=1                           # over relaxation factor
cr=1e-8                       # convergence criteria for steady state condition
r= (dx/dy)**2

q=np.zeros(2)
q[0]=input(' choose differencing scheme (1,2 or 3):\n 1-upwind\n 2-central\n 3-hybrid\n')
q[1]=input('\n choose method of solving the system of equations (1,2 or 3):\n 1-PSOR\n 2-LSOR (y sweep)\n 3-LSOR (x sweep)\n')

s1=["upwind","central","hybrid"]
s2=["PSOR","LSOR (y sweep)","LSOR (x sweep)"]

# calculating velocity field
VELOCITY(u,v)
  
# initialization
INITIALIZE(T)
        
# boundary conditions
BC(T)
 
# determining differencing scheme
if q[0]==1:

    
    # upwind scheme
    UPWIND(a_w,a_e,a_s,a_n,a_p,u,v,r)   
    
elif q[0]==2:  
     # centeral scheme
    CENTRAL(a_w,a_e,a_s,a_n,a_p,u,v,r)
          
elif q[0]==3:     

    # hybrid scheme
    HYBRID(a_w,a_e,a_s,a_n,a_p,u,v,r)


#================================== main loop =================================
tic=time.time()      
k=0
L2=1
while L2>cr:
    k=k+1
    
    T_old[:,:]=T[:,:]
    
    # determining method of solving the system of equations
    if q[1]==1:         
        PSOR(T,a_w,a_e,a_s,a_n)            
    
    elif q[1]==2:
        LSOR_y(T,a_w,a_e,a_s,a_n)
    
    elif q[1]==3:
        LSOR_x(T,a_w,a_e,a_s,a_n)
  
    # updating boundary conditions
    BC(T)  
        
    # calculating L2 norm of the error
    L2=NORM(T,T_old)
    
    # saving temperature at the middle of the channel
    e[k]=T[int(l/(2*dx)),int(h/(2*dy))]
                
    # printing iteration and error
    print('{:10d}{:15.2e}'.format(k,L2))
    

toc=time.time()      
#================================== results ===================================
print('\n==================================================================\n')
print(' calculation completed\n')
print(' execution time:{:11.3f}s'.format(toc-tic))
print(' mesh:{:18d}*{:2d}'.format(m,n))
print(" differencing scheme: {:s}".format(s1[int(q[0]-1)]))
print(" method of solving the system of equations: {:s}".format(s2[int(q[1]-1)]))
i=int(l/(2*dx))
j=int(h/(2*dy))
print(' number of iterations until convergence: {:6d}'.format(k))
print(' temperature at the middle of the channel: {:8.4f}\n'.format(T[i,j]))

# generating the grid
Y,X=np.meshgrid(np.linspace(0,h,n+1),np.linspace(0,l,m+1))

# temperature contour
fig1=plt.figure(1)
plt.contourf(X,Y,T,50,cmap='jet')
plt.axes().set_aspect('equal')
plt.xlabel('x(m)');plt.ylabel('y(m)')

# temperature surface
fig2=plt.figure(2)
ax=plt.axes(projection='3d')
ax.plot_surface(X,Y,T,cmap='jet')
ax.set_xlabel('x(m)');ax.set_ylabel('y(m)');ax.set_zlabel('T($^oC$)')
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 0.166, 0.5, 0.7]))

# temperature at the vertical middle line
fig3=plt.figure(3)
plt.plot(T[i,:],np.linspace(0,h,n+1))
plt.xlabel('T($^oC$)');plt.ylabel('y(m)')
plt.ylim(0,h)

# temperature at the horizontal middle line
fig4=plt.figure(4)
plt.plot(np.linspace(0,l,m+1),T[:,j])
plt.xlabel('x(m)');plt.ylabel('T($^oC$)')
plt.xlim(0,l)

# temperature vs iteration
fig5=plt.figure(5)
plt.plot(np.linspace(1,k,k-1),e[1:k])
plt.xlabel('iteration');plt.ylabel('T($^oC$)')
plt.xlim(0,k)
plt.show()