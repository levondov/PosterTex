
import Image
import numpy as np
import matplotlib.pyplot as plt

A1 = np.genfromtxt("verstreakonly.txt", dtype=None)
A2 = np.genfromtxt("lowalpha_1uA.txt", dtype=None)
A3 = np.genfromtxt("analog10sx50_backgroundsubstracted.txt", dtype=None)
A4 = np.genfromtxt("lowalpha_100mA_zerisseneFuellungmitLuecke.txt", dtype=None)

im1 = Image.fromarray(np.uint8(plt.cm.flag(A1)*255))
#im1new = im1.resize((500,500),Image.ANTIALIAS)
im2 = Image.fromarray(np.uint8(A2*255))
im3 = Image.fromarray(np.uint8(A3*255))
im4 = Image.fromarray(np.uint8(A4*255))

hist1 = im1.histogram()
hist2 = im2.histogram()
hist3 = im3.histogram()
hist4 = im4.histogram()

plt.plot(hist1)
plt.show()

#im1.save('verstreakonly.png')
#im2.save('lowalpha_1uA.png')
#im3.save('analog10sx50_backgroundsubstracted.png')
#im4.save('lowalpha_100mA_zerisseneFuellungmitLuecke.png')

fig1 = plt.figure()
fig1.set_size_inches(13, 13)
fig1.figimage(im1)
#plt.show()
plt.savefig('verstreakonly.pdf',dpi=75)

fig2 = plt.figure()
fig2.set_size_inches(13, 13)
fig2.figimage(im2)
#plt.show()
plt.savefig('lowalpha_1uA.pdf',dpi=100)

fig3 = plt.figure()
fig3.set_size_inches(13, 13)
fig3.figimage(im3)
#plt.show()
plt.savefig('analog10sx50_backgroundsubstracted.pdf',dpi=75)

fig4 = plt.figure()
fig4.set_size_inches(13, 13)
fig4.figimage(im4)
#plt.show()
plt.savefig('lowalpha_100mA_zerisseneFuellungmitLuecke.pdf',dpi=100)
