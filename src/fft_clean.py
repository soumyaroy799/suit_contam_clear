import os, sys
from sunpy.map import Map, MapSequence
import matplotlib.pyplot as plt 
import time
import threading
import glob
import numpy as np

path = '/home/sroy/suit_contam_clear/data/'

f = sorted(glob.glob(f'{path}*NB07*'))

a = Map(f[3])

f = np.fft.fft2(a.data)
fshift = np.fft.fftshift(f)

rows, cols = a.data.shape
crow, ccol = rows // 2, cols // 2

sigma = 55
y, x = np.ogrid[:rows, :cols]
dist2 = (x - ccol)**2 + (y - crow)**2
gaussian_mask = np.exp(-dist2 / (2*sigma**2))

fshift_filtered = fshift * gaussian_mask

f_ishift = np.fft.ifftshift(fshift_filtered)
filtered = np.fft.ifft2(f_ishift)
filtered = np.real(filtered)  # drop tiny imaginary parts

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
c = axs[0].imshow(a.data, cmap='gray', origin='lower')
cbar = plt.colorbar(c)
cbar.remove()
axs[0].set_title("Original")
c = axs[1].imshow(filtered, cmap='gray', origin='lower')
cbar = plt.colorbar(c)
cbar.remove()
axs[1].set_title("Filtered")
d = (a.data-filtered)/a.data
d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
# c = axs[2].imshow(d, origin='lower', cmap='seismic', vmin=-1500, vmax=1500)
c = axs[2].imshow(d, origin='lower', cmap='seismic', vmin=-.3, vmax=.3)
plt.colorbar(c, fraction=0.046, pad=0.04)
plt.show()

plt.hist(d.ravel(), bins=500); plt.xlim(-0.2, 0.2); plt.show()