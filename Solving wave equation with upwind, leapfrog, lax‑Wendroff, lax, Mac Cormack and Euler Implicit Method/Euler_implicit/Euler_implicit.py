"""
This code was my university project 
Code by: Reza Lotfi Navaei

"""
#===================================
#           Euler_implicit
#===================================
import math
import numpy as np
import matplotlib.pyplot as plt


#========================= function definition =========================
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
    


#============================= main program ============================
m=100                          # number of horizontal divisions     
n=50                           # number of time interval divisions  

# matrix definition
u=np.zeros((m+1,n+1))
u_exact=np.zeros(m+1)
x=np.zeros(m+1)
a=np.zeros(m-1);b=np.zeros(m-1);c=np.zeros(m-1);r=np.zeros(m-1);s=np.zeros(m-1)

l=40.                         # length of the domain
Tf=20.                        # final time
dx=l/m
dt=Tf/n
ve=1.                         # wave velocity
v=ve*dt/dx                    # Courant number
pi=math.acos(-1)              # pi number
k=2.

# generating the grid
for i in range(m+1):
    x[i]=i*dx

# initial condition
for i in range(m+1):
    u[i,0]=math.sin(2*k*pi*x[i]/l)
    
    
    
#=============================== solution ==============================
for j in range(1,n+1):

    # updating boundary conditions
    u[0,j]=u[0,j-1]-v*(u[0,j-1]-u[m-1,j-1])
    u[m,j]=u[0,j]

    for i in range(1,m):
        a[i-1]=v/2
        b[i-1]=-1
        c[i-1]=-v/2
        r[i-1]=-u[i,j-1]
    
    i=1    
    r[i-1]=r[i-1]-a[i-1]*u[i-1,j]
    i=m-1    
    r[i-1]=r[i-1]-c[i-1]*u[i+1,j] 
    
    TDMA(a,b,c,r,s)
    
    for i in range(1,m):
        u[i,j]=s[i-1]



#=============================== results ===============================
# calculating the exact solution
for i in range(m+1):
    u_exact[i]=math.sin(2*k*pi*(x[i]-ve*Tf)/l)
    
print('\nCourant number: {:5.3f}'.format(v))

# printing u at the middle of the domain
i=int(l/(2*dx))
print('u at the middle of the domain: {:5.5f}\n'.format(u[i,n])) 

# ploting the numerical solution
plt.figure()
plt.plot(x,u[:,n])
plt.xlabel('x');plt.ylabel('u')
plt.xlim(0,l);plt.ylim(-1,1)
plt.title('numerical solution')

# ploting the exact solution
plt.figure()
plt.plot(x,u_exact)
plt.xlabel('x');plt.ylabel('u')
plt.xlim(0,l);plt.ylim(-1,1)
plt.title('exact solution')

# animation
plt.ion()
fig,ax=plt.subplots()
line,=ax.plot(x,u[:,0])
plt.xlim(0,l);plt.ylim(-1,1)
plt.title('numerical solution')

for j in range(n+1):
        
    line.set_ydata(u[:,j])
    fig.canvas.draw() 
    plt.pause(0.1)
    
input("Press Enter to continue... ")