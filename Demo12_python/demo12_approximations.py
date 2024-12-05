import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from scipy.fft import fft, fftshift, ifft, dct, idct

from perform_thresholding import perform_thresholding
from fwt import fwt
from ifwt import ifwt
from general import reverse

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% APPROXIMATIONS LIN�AIRES DANS DES BASES
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = loadmat('piece_regular.mat')['piece_regular']
f = np.squeeze(f)
n = 512
m = 128   # nombre de coefficients qui gardent

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Approximation dans Fourier
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ft = fftshift(fft(f))
ft_a = np.zeros_like(f, dtype=complex)
ft_b = np.zeros_like(f, dtype=complex)

sel = [int(n/2-m/2), int(n/2+m/2)]
ft_a[sel[0]:sel[1]] = ft[sel[0]:sel[1]]    # Approximation lineaire gardant que les 128 premiers coeffs
ft_b = perform_thresholding(ft, m, 'largest') # approximation non-lineare prenant que les m plus grands coeffcients

f1 = ifft(fftshift(ft))
fm = np.abs(ifft(fftshift(ft_a)));
fn = np.abs(ifft(fftshift(ft_b)))

fig, axs = plt.subplots(4,1, constrained_layout=True)
fig.set_size_inches(15, 8)
plt.suptitle('Fourier', fontsize=16)
axs[0].plot(f)
axs[0].set_title('Signal Original')
axs[0].axes.get_xaxis().set_visible(False)
axs[0].axes.get_yaxis().set_visible(False)

axs[1].plot(f1)
axs[1].set_title('\n\n Signal avec Fourier au complet \n Erreur relarive = ' + str(np.linalg.norm(f-f1)/np.linalg.norm(f)))
axs[1].axes.get_xaxis().set_visible(False)
axs[1].axes.get_yaxis().set_visible(False)

axs[2].plot(fm)
axs[2].set_title('\n\n Approximation linéaire avec ' + str(m) + ' coefficients FFT \n Erreur relarive = ' + str(np.linalg.norm(f-fm)/np.linalg.norm(f)))
axs[2].axes.get_xaxis().set_visible(False)
axs[2].axes.get_yaxis().set_visible(False)

axs[3].plot(fn)
axs[3].set_title('\n\n Approximation non-linéaire avec ' + str(m) + ' coefficients FFT \n Erreur relarive = ' + str(np.linalg.norm(f-fn)/np.linalg.norm(f)))
axs[3].axes.get_xaxis().set_visible(False)
axs[3].axes.get_yaxis().set_visible(False)
plt.savefig('fft.png')
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Approximation avec des cosinus discrets
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fc = dct(f) 
fc_a = np.zeros_like(f)
fc_b = np.zeros_like(f)
fc_a[0:m] = fc[0:m]    #% Approximation linéaire gardant que les 128 premiers coeffs
fc_b = perform_thresholding(fc, m, 'largest') # approximation non-lineare prenant que les m plus grands coeffcients

f1 = idct(fc)
fm = idct(fc_a)
fn = idct(fc_b)

fig, axs = plt.subplots(4,1, constrained_layout=True)
fig.set_size_inches(15, 8)
plt.suptitle('Fourier', fontsize=16)
axs[0].plot(f)
axs[0].set_title('Signal Original')
axs[0].axes.get_xaxis().set_visible(False)
axs[0].axes.get_yaxis().set_visible(False)

axs[1].plot(f1)
axs[1].set_title('\n\n Signal avec DCT au complet \n Erreur relarive = ' + str(np.linalg.norm(f-f1)/np.linalg.norm(f)))
axs[1].axes.get_xaxis().set_visible(False)
axs[1].axes.get_yaxis().set_visible(False)

axs[2].plot(fm)
axs[2].set_title('\n\n Approximation linéaire avec ' + str(m) + ' coefficients DCT \n Erreur relarive = ' + str(np.linalg.norm(f-fm)/np.linalg.norm(f)))
axs[2].axes.get_xaxis().set_visible(False)
axs[2].axes.get_yaxis().set_visible(False)

axs[3].plot(fn)
axs[3].set_title('\n\n Approximation non-linéaire avec ' + str(m) + ' coefficients DCT \n Erreur relarive = ' + str(np.linalg.norm(f-fn)/np.linalg.norm(f)))
axs[3].axes.get_xaxis().set_visible(False)
axs[3].axes.get_yaxis().set_visible(False)
plt.savefig('dct.png')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Approximation avec des cosinus locaux
#% 
#% JPEG est basé sur cette base
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# taille du segment
w = 32

# coefficients de la DCT locale full, lineare et non-lineare
fcl = np.zeros_like(f)
fcl_a = np.zeros_like(f)
fcl_b = np.zeros_like(f)
local_segment = np.zeros_like(f)

# DCT locale full
for i in range(int(n/w)):
    seli_full = np.array([i * w, i * w + w])
    fcl[seli_full[0]:seli_full[1]] = dct(f[seli_full[0]:seli_full[1]])
    
# On a 32 segments de taille 32 (1024 / 32 = 32). Donc, si on garde 128 coefficients au total,
# on garde que s coefficients (128/32=4) par segment. 
s = int(m / (n / w))
for i in range(int(n/w)):
    seli_full = np.array([i * w, i * w + w])
    # que les s premiers indices du segment i
    seli_cut = np.array([i * w, i * w + s])

    # la full DCT local prends tous les coefficients s a chaque segment 
    f1[seli_full[0]:seli_full[1]] = idct(fcl[seli_full[0]:seli_full[1]])

    # l'approximation lineaire garde que les s premiers coefficients
    # le reste des coefficients restent à zéro
    local_segment[seli_cut[0]:seli_cut[1]] = fcl[seli_cut[0]:seli_cut[1]]
    fm[seli_full[0]:seli_full[1]] = idct(local_segment[seli_full[0]:seli_full[1]])

fcl_b = perform_thresholding(fcl, m, 'largest')
for i in range(int(n/w)):
    seli_full = np.array([i * w, i * w + w])
    fn[seli_full[0]:seli_full[1]] = idct(fcl_b[seli_full[0]:seli_full[1]])

fig, axs = plt.subplots(4,1, constrained_layout=True)
fig.set_size_inches(15, 8)
plt.suptitle('DCT locales', fontsize=16)
axs[0].plot(f)
axs[0].set_title('Signal Original')
axs[0].axes.get_xaxis().set_visible(False)
axs[0].axes.get_yaxis().set_visible(False)

axs[1].plot(f1)
axs[1].set_title('\n\n Signal avec la DCT locale au complet \n Erreur relarive = ' + str(np.linalg.norm(f-f1)/np.linalg.norm(f)))
axs[1].axes.get_xaxis().set_visible(False)
axs[1].axes.get_yaxis().set_visible(False)

axs[2].plot(fm)
axs[2].set_title('\n\n Approximation linéaire avec ' + str(m) + ' coefficients DCT locale \n Erreur relarive = ' + str(np.linalg.norm(f-fm)/np.linalg.norm(f)))
axs[2].axes.get_xaxis().set_visible(False)
axs[2].axes.get_yaxis().set_visible(False)

axs[3].plot(fn)
axs[3].set_title('\n\n Approximation non-linéaire avec ' + str(m) + ' coefficients DCT locale \n Erreur relarive = ' + str(np.linalg.norm(f-fn)/np.linalg.norm(f)))
axs[3].axes.get_xaxis().set_visible(False)
axs[3].axes.get_yaxis().set_visible(False)
plt.savefig('dct_locale.png')
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Approximation avec des ondelettes de Haar, Daubechies et autres
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fw = np.zeros_like(f)
fw_a = np.zeros_like(f)
fw_b = np.zeros_like(f)

# On commence avec Haar = D2 
h = [0, 1/np.sqrt(2), 1/np.sqrt(2)]
g = [0, 1/np.sqrt(2), -1/np.sqrt(2)]

fw = fwt(f, h, g, visu=False)
f1 = ifwt(fw, reverse(h), reverse(g), visu=False)

fw_a[0:m] = fw[0:m]
fw_b = perform_thresholding(fw, m, 'largest')

fm = ifwt(fw_a, reverse(h), reverse(g), visu=False)
fn = ifwt(fw_b, reverse(h), reverse(g), visu=False)
fig, axs = plt.subplots(4,1, constrained_layout=True)
fig.set_size_inches(15, 8)
plt.suptitle('Ondelettes de Haar', fontsize=16)
axs[0].plot(f)
axs[0].set_title('Signal Original')
axs[0].axes.get_xaxis().set_visible(False)
axs[0].axes.get_yaxis().set_visible(False)

axs[1].plot(f1)
axs[1].set_title('\n\n Signal avec Haar au complet \n Erreur relarive = ' + str(np.linalg.norm(f-f1)/np.linalg.norm(f)))
axs[1].axes.get_xaxis().set_visible(False)
axs[1].axes.get_yaxis().set_visible(False)

axs[2].plot(fm)
axs[2].set_title('\n\n Approximation linéaire avec ' + str(m) + ' coefficients de Haar \n Erreur relarive = ' + str(np.linalg.norm(f-fm)/np.linalg.norm(f)))
axs[2].axes.get_xaxis().set_visible(False)
axs[2].axes.get_yaxis().set_visible(False)

axs[3].plot(fn)
axs[3].set_title('\n\n Approximation non-linéaire avec ' + str(m) + ' coefficients de Haar \n Erreur relarive = ' + str(np.linalg.norm(f-fn)/np.linalg.norm(f)))
axs[3].axes.get_xaxis().set_visible(False)
axs[3].axes.get_yaxis().set_visible(False)
plt.savefig('haar.png')
plt.show()


fw_d4 = np.zeros_like(f)
fw_a_d4 = np.zeros_like(f)
fw_b_d4 = np.zeros_like(f)

# Daubechies = D4 
h = [ 0,    0.4830,    0.8365,    0.2241,   -0.1294 ]
g = [ 0,   -0.1294,   -0.2241,    0.8365,   -0.4830 ]

fw_d4 = fwt(f, h, g, visu=False)
f1 = ifwt(fw_d4, reverse(h), reverse(g), visu=False)

fw_a_d4[0:m] = fw_d4[0:m]
fw_b_d4 = perform_thresholding(fw_d4, m, 'largest')

fm = ifwt(fw_a_d4, reverse(h), reverse(g), visu=False)
fn = ifwt(fw_b_d4, reverse(h), reverse(g), visu=False)

fig, axs = plt.subplots(4,1, constrained_layout=True)
fig.set_size_inches(15, 8)
plt.suptitle('Ondelettes D4', fontsize=16)
axs[0].plot(f)
axs[0].set_title('Signal Original')
axs[0].axes.get_xaxis().set_visible(False)
axs[0].axes.get_yaxis().set_visible(False)

axs[1].plot(f1)
axs[1].set_title('\n\n Signal avec D4 au complet \n Erreur relarive = ' + str(np.linalg.norm(f-f1)/np.linalg.norm(f)))
axs[1].axes.get_xaxis().set_visible(False)
axs[1].axes.get_yaxis().set_visible(False)

axs[2].plot(fm)
axs[2].set_title('\n\n Approximation linéaire avec ' + str(m) + ' coefficients de D4 \n Erreur relarive = ' + str(np.linalg.norm(f-fm)/np.linalg.norm(f)))
axs[2].axes.get_xaxis().set_visible(False)
axs[2].axes.get_yaxis().set_visible(False)

axs[3].plot(fn)
axs[3].set_title('\n\n Approximation non-linéaire avec ' + str(m) + ' coefficients de D4 \n Erreur relarive = ' + str(np.linalg.norm(f-fn)/np.linalg.norm(f)))
axs[3].axes.get_xaxis().set_visible(False)
axs[3].axes.get_yaxis().set_visible(False)
plt.savefig('D4.png')
plt.show()

fig, axs = plt.subplots(5,1, constrained_layout=True)
fig.set_size_inches(15, 8)
plt.suptitle('Coefficients de toutes les transformées', fontsize=16)
axs[0].plot(np.log(np.abs(ft)))
axs[0].set_title('FFT')
axs[0].set_ylim([0, 6])
axs[1].plot(np.log(np.abs(fc)))
axs[1].set_title('DCT')
axs[1].set_ylim([0, 6])
axs[2].plot(np.abs(fcl))
axs[2].set_title('DCT locale')
axs[2].set_ylim([0, 6])
axs[3].plot(np.abs(fw))
axs[3].set_title('Ondelettes de Haar')
axs[3].set_ylim([0, 6])
axs[4].plot(np.abs(fw_d4))
axs[4].set_title('Ondelettes de D4')
axs[4].set_ylim([0, 6])
plt.savefig('coeffs_full.png')
plt.show()

fig, axs = plt.subplots(5,1, constrained_layout=True)
fig.set_size_inches(15, 8)
plt.suptitle('Coefficients de toutes les transformées non-linéaires', fontsize=16)
axs[0].plot(np.log(np.abs(ft_b)))
axs[0].set_title('FFT non-linéaire')
axs[0].set_ylim([0, 6])
axs[1].plot(np.log(np.abs(fc_b)))
axs[1].set_title('DCT non-linéaire')
axs[1].set_ylim([0, 6])
axs[2].plot(np.abs(fcl_b))
axs[2].set_title('DCT locale non-linéaire')
axs[2].set_ylim([0, 6])
axs[3].plot(np.abs(fw_b))
axs[3].set_title('Ondelettes de Haar non-linéaire')
axs[3].set_ylim([0, 6])
axs[4].plot(np.abs(fw_b_d4))
axs[4].set_title('Ondelettes de D4 non-linéaire')
axs[4].set_ylim([0, 6])
plt.savefig('coeffs_nonlin.png')
plt.show()

