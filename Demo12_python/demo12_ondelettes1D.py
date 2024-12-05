import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from perform_thresholding import perform_thresholding
from compute_wavelet_filter import compute_wavelet_filter
from cconv import cconv
from general import upsampling, subsampling, reverse, circshift1d
from fwt import fwt
from ifwt import ifwt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = loadmat('piece_regular.mat')['piece_regular']
f = np.squeeze(f)
n = 512

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Approximation avec des ondelettes
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Haar = D2
h = [0, 1/np.sqrt(2), 1/np.sqrt(2)]
g = [0, 1/np.sqrt(2), -1/np.sqrt(2)]
h_d2 = compute_wavelet_filter('Daubechies', 1) # Haar = D2 = Daubechies d'ordre 1, 2 coefficients non nul
print(h, h_d2, g)

# convolution spherique sur la dimension 0, necessaire pour les ondelettes
approx = cconv( f, h_d2, 0)
a = subsampling(approx)
print('Dimension de la cconv(f) et sa version subsampled par un facteur 2', approx.size, a.size)
plt.plot(approx, 'b')
plt.plot(a, 'r')
plt.show()

# On peut donc faire des ondelettes avec la convolution circulaire et des subsampling
a = subsampling( cconv( f, h_d2, 0) ) # approximations
d = subsampling( cconv( f, g, 0) )    # details
plt.plot(np.concatenate((a, d)), 'b')
plt.show()

# on revient en arriere
reconstruction = cconv(upsampling(a), reverse(h_d2), 0) + cconv(upsampling(d), reverse(g), 0)
plt.title('Signal original et reconstruction en ondelettes avec erreur ' + str(np.linalg.norm(f-reconstruction)/np.linalg.norm(f)))
plt.plot(f, 'b')
plt.plot(reconstruction, 'r.-')
plt.show()

fw = fwt(f, h, g, visu=True)
f1 = ifwt(fw, reverse(h), reverse(g), visu=True)

fig, axs = plt.subplots(2,1, constrained_layout=True)
fig.set_size_inches(15, 8)
plt.suptitle('Ondelette de Haar', fontsize=16)
axs[0].plot(f)
axs[0].set_title('Signal Original')
axs[0].axes.get_xaxis().set_visible(False)
axs[0].axes.get_yaxis().set_visible(False)
axs[1].plot(f1)
axs[1].set_title('\n\n Signal avec ondelettes de Haar au complet \n Erreur relarive = ' + str(np.linalg.norm(f-f1)/np.linalg.norm(f)))
axs[1].axes.get_xaxis().set_visible(False)
axs[1].axes.get_yaxis().set_visible(False)
plt.show()


#% D4 - Ondelettes de Daubechies 4
h = [ 0,    0.4830,    0.8365,    0.2241,   -0.1294 ]
g = [ 0,   -0.1294,   -0.2241,    0.8365,   -0.4830 ]
h_d4 = compute_wavelet_filter('Daubechies', 4) # D4 = Daubechies d'ordre 2 => 4 coefficients non nul
print(h, h_d4, g)

fw = fwt(f, h, g)
f1 = ifwt(fw, reverse(h), reverse(g))

fig, axs = plt.subplots(2,1, constrained_layout=True)
fig.set_size_inches(15, 8)
plt.suptitle('Ondelette de Daubechies 4 (D4)', fontsize=16)
axs[0].plot(f)
axs[0].set_title('Signal Original')
axs[0].axes.get_xaxis().set_visible(False)
axs[0].axes.get_yaxis().set_visible(False)
axs[1].plot(f1)
axs[1].set_title('\n\n Signal avec ondelettes D4 au complet \n Erreur relarive = ' + str(np.linalg.norm(f-f1)/np.linalg.norm(f)))
axs[1].axes.get_xaxis().set_visible(False)
axs[1].axes.get_yaxis().set_visible(False)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Forme des ondelettes
#
# Un peu comme avec Fourier, si on met des zeros dans le monde des
# ondelettes et qu'on fait une ifwt, on pourra visualiser l'ondelette
# (comme l'exercise des COS en fourier)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
JMax = int(np.log2(n))
k = 0
for j in np.arange(JMax):
    print('Échelle ou résolution j = ', j)
    fw = np.zeros(n) # spectre d'ondelettes vide
    fw[2**j] = 1     # on met un 1 à la résolution ou échelle j, rien ailleurs
    f1 = ifwt(fw, reverse(h), reverse(g))
    f1 = circshift1d(f1, int(n/2))
    if j == 0 :
        plt.title('Forme de l\'ondelette mère', fontsize=16)
    else :
        plt.title('Forme des ondelettes à la résolution j = ' + str(j), fontsize=16)
    plt.plot(f1)
    plt.show()
    k = k+1;    


# on refait les formes pour Haar
h = [0, 1/np.sqrt(2), 1/np.sqrt(2)]
g = [0, 1/np.sqrt(2), -1/np.sqrt(2)]
JMax = int(np.log2(n))
k = 0
for j in np.arange(JMax):
    print('Échelle ou résolution j = ', j)
    fw = np.zeros(n) # spectre d'ondelettes vide
    fw[2**j] = 1     # on met un 1 à la résolution ou échelle j, rien ailleurs
    f1 = ifwt(fw, reverse(h), reverse(g))
    f1 = circshift1d(f1, int(n/2))
    if j == 0 :
        plt.title('Forme de l\'ondelette mère', fontsize=16)
    else :
        plt.title('Forme des ondelettes à la résolution j = ' + str(j), fontsize=16)
    plt.plot(f1)
    plt.show()
    k = k+1;    
