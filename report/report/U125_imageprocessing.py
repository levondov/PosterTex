import Image
import numpy as np
import scipy.stats as ss
from scipy import optimize
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


A1 = np.loadtxt("U125_Harmonics.txt", dtype=None)

plt.plot(A1[:,0],A1[:,1],'k')
plt.grid(True)
plt.xlabel('wavelength (nm)')
plt.ylabel('normalized detector signal (a.u.)')
plt.savefig('uplot.pdf',transparent=True)
plt.show()