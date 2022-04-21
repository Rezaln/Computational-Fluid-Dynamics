"""
This code was my university project 
Code by: Reza Lotfi Navaei

"""
#===================================
#               upwind
#===================================
import math
import numpy as np
import matplotlib.pyplot as plt

m=100                          # number of horizontvel divisions     
n=50                           # number of time interval divisions  

# matrix definition
u=np.zeros((m+1,n+1))
u_exact=np.zeros(m+1)
x=np.zeros(m+1)

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

    for i in range(1,m):
        u[i,j]=u[i,j-1]-v*(u[i,j-1]-u[i-1,j-1])
        
    # updating boundary conditions
    u[0,j]=u[0,j-1]-v*(u[0,j-1]-u[m-1,j-1])
    u[m,j]=u[0,j]
    
    # printing time
    print('{:10.2f}'.format(j*dt))



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