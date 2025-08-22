import matplotlib.pyplot as plt
import numpy as np

from fonctions.io import read_data
from fonctions.ondelette import computeHaar1D, computeHaar1DReconstruction
from fonctions.threshold import perform_thresholding

#### Demo des ondelettes de Haar 1D
############# 1 D Haar Wavelet DECOMPOSITION ################
# get signal
f = read_data('piece-regular_512.npy')
n = 512
plt.plot(f)
plt.show()

# Initialize coeffs as signal itself,
# scale 2^9 = 512 bins, resolution 2^{-9}
fw = f.copy()

# initial scale j is the max one
j = np.log2(n) - 1

# fw coefficients will be iteratively computed until 
# they contain the Haar wavelet
A = fw[0:2**(int(j)+1)]

# 256 x 1
# We compute average and differences to get coarse and details.
# They are weighted by 1/sqrt(2) to maintain orthogonality.

Coarse = (A[::2] + A[1::2]) / 2
Detail = (A[::2] - A[1::2]) / 2

A = np.concatenate((Coarse, Detail))

fig, axs = plt.subplots(2, 1)
axs[0].plot(f, '.-')
axs[0].set_title('Signal')
axs[1].plot(Coarse, '.-')
axs[1].set_title('Coarse')
plt.show()

# On zoom dans une region de Coarse
s = 400
t = 40
fig, axs = plt.subplots(2, 1)
axs[0].plot(f, 'o-')
axs[0].set_xlim([s-t, s+t])
axs[0].set_ylim([0, 1])
axs[0].set_title('Signal (zoom)')
axs[1].plot(Coarse, 'o-')
axs[1].set_xlim([(s-t)/2, (s+t)/2])
axs[1].set_ylim([np.min(A), np.max(A)])
axs[1].set_title('Averages (zoom)')
plt.show()

# plot Details
fig, axs = plt.subplots(2, 1)
axs[0].plot(f, '.-')
axs[0].set_title('Signal')
axs[1].plot(Detail, '.-')
axs[1].set_title('Details')
plt.show()

# plot both
fig, axs = plt.subplots(2, 1)
axs[0].plot(f)
axs[0].set_title('Signal')
axs[1].plot(A)
axs[1].set_title('Transformed')
plt.show()

fig, axs = plt.subplots(2, 1)
axs[0].plot(A[0:256])
axs[0].set_title('Signal')
axs[1].plot(A[256:])
axs[1].set_title('Transformed')
plt.show()

#################################################################
fw = computeHaar1D(f, n)
#################################################################
plt.plot(fw)
plt.show()

print('Energy of the signal       = ', np.linalg.norm(f) ** 2)
print('Energy of the coefficients = ', np.linalg.norm(fw) ** 2)

########################################################################
# 1D Reconstruction
#
# Initialize the image to recover f1 as the transformed coefficient,
# and select the smallest possible scale.
########################################################################

f1 = fw.copy()
j = 0 

# Retrieve coarse and detail coefficients in the vertical direction.
Coarse = f1[:2**j].copy()
Detail = f1[2**j:2**(j+1)].copy()

f1[0:2**(j+1):2] = (Coarse + Detail) / np.sqrt(2)
f1[1:2**(j+1):2] = (Coarse - Detail) / np.sqrt(2)

f1 = computeHaar1DReconstruction(fw, n)

# Original image versus Reconstructed image
print('Error |f-f1|/|f| = ', np.linalg.norm(f-f1)/np.linalg.norm(f))

fig, axs = plt.subplots(2, 1)
axs[0].plot(f)
axs[0].set_title('Original Signal')
axs[1].plot(f1)
axs[1].set_title('Reconstructed signal with full HAAR transform, error = ' + str(np.linalg.norm(f-f1)/np.linalg.norm(f)))
plt.show()

##################
# Approximations #
##################
cut = 100
fw_a = np.zeros((fw.shape[0], 1))
fw_a[0:cut] = fw[0:cut]

fw_max = perform_thresholding(fw, cut, 'largest')

f1_a = computeHaar1DReconstruction(fw_a, n)
f1_max = computeHaar1DReconstruction(fw_max, n)

fig, axs = plt.subplots(3, 1)
axs[0].plot(f)
axs[0].set_title('Original Signal')
axs[1].plot(f1_a)
axs[1].set_title('Linear approx with ' + str(cut) + 'coefficients')
print('Error |f-f1_a|/|f| = ', np.linalg.norm(f-f1_a)/np.linalg.norm(f))
axs[2].plot(f1_max)
axs[2].set_title('Nonlinear approx with ' + str(cut) + 'coefficients')
print('Error |f-f1_max|/|f| = ', np.linalg.norm(f-f1_max)/np.linalg.norm(f))
plt.show()
