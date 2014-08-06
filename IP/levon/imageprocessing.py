
import Image
import numpy as np
import scipy.stats as ss
from scipy import optimize
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

A1 = np.loadtxt("verstreakonly.txt", dtype=None)
A2 = np.loadtxt("lowalpha_1uA.txt", dtype=None)
A3 = np.loadtxt("analog10sx50_backgroundsubstracted.txt", dtype=None)
A4 = np.loadtxt("lowalpha_100mA_zerisseneFuellungmitLuecke.txt", dtype=None)

A1 = np.delete(A1,0,axis=0)
axis1 = np.fliplr([A1[:,0]])
axis1 = axis1[0,:]
A1 = np.delete(A1,0,axis=1)
sumA1 = np.sum(A1,axis=1)

A2 = np.delete(A2,0,axis=0)
axis2 = np.fliplr([A2[:,0]])
axis2 = axis2[0,:]
data = A2
A2 = np.delete(A2,0,axis=1)
sumA2 = np.sum(A2,axis=1)

A3 = np.delete(A3,0,axis=0)
axis3 = np.fliplr([A3[:,0]])
axis3 = axis3[0,:]
A3 = np.delete(A3,0,axis=1)
sumA3 = np.sum(A3,axis=1)

A4 = np.delete(A4,0,axis=0)
axis4 = np.fliplr([A4[:,0]])
axis4 = axis4[0,:]
A4 = np.delete(A4,0,axis=1)
sumA4 = np.sum(A4,axis=1)



np.savetxt('axis2',axis2)
np.savetxt('sumA2',sumA2)

im1 = Image.fromarray(np.uint8(A1))
im2 = Image.fromarray(np.uint8(A2))
im3 = Image.fromarray(np.uint8(A3))
im4 = Image.fromarray(np.uint8(A4))

a1 = np.linalg.norm(sumA1)
b1 = sumA1/a1

fig1 = plt.figure()
plt.imshow(im1)
plt.axis('off')
plt.savefig('verstreakonly.pdf',dpi=100,transparent=True)


fig11 = plt.figure()
plt.plot(b1,axis1,'w--',linewidth=4)
plt.grid(True)
plt.axis([0.02,0.4,-105,280])
plt.xlabel('intensity (a.u.)')
plt.ylabel('time range (ps) ')
plt.savefig('verstreakonlygraph.pdf',transparent=True)

a2 = np.linalg.norm(sumA2)
b2 = sumA2/a2

fig2 = plt.figure()
plt.imshow(im2)#,cmap=plt.cm.prism,vmin=-20,vmax=75)
plt.axis('off')
plt.savefig('lowalpha_1uA.pdf',dpi=200,transparent=True)

fig22 = plt.figure()
plt.xscale('log')
plt.plot(sumA2,axis2,'w--',linewidth=4)
plt.grid(True)
plt.axis([30000,100000000,30,120])
plt.xlabel('intensity (a.u.)')
plt.ylabel('time range (ps) ')
plt.savefig('lowalpha_1uAgraph.pdf',transparent=True)
print(np.shape(sumA2),np.shape(axis2))
mu, stn = norm.fit(np.array([sumA2,axis2]))
print(mu,stn)
data = norm.rvs(10.0, 2.5, size=100)
print(data)

a3 = np.linalg.norm(sumA3)
b3 = sumA3/a3

fig3 = plt.figure()
plt.imshow(im3)
plt.axis('off')
plt.savefig('analog10sx50_backgroundsubstracted.pdf',dpi=100,transparent=True)

fig33 = plt.figure()
plt.plot(b3,axis3,'w--',linewidth=4)
plt.grid(True)
plt.axis([0.05,0.4,20,180])
plt.xlabel('intensity (a.u.)')
plt.ylabel('time range (ps) ')
plt.savefig('analog10sx50_backgroundsubstractedgraph.pdf',transparent=True)

a4 = np.linalg.norm(sumA4)
b4 = sumA4/a4

fig4 = plt.figure()
plt.imshow(im4)
plt.axis('off')
plt.savefig('lowalpha_100mA_zerisseneFuellungmitLuecke.pdf',dpi=100,transparent=True)


fig44 = plt.figure()
plt.xscale('log')
plt.plot(sumA4,axis4,'w--',linewidth=4)
plt.grid(True)
plt.axis([30000,100000000,100,315])
plt.xlabel('intensity (a.u.)')
plt.ylabel('time range (ps) ')
plt.savefig('lowalpha_100mA_zerisseneFuellungmitLueckegraph.pdf',transparent=True)

#plt.show()