import matplotlib.pyplot as plt
import numpy as np

t = np.arange(-150, 150, 0.01)

y1 = np.cos(t / 3)
y2 = np.cos(t / 4)

ys = np.cos(t / 3) + np.cos(t / 4)

plt.plot(t, y1, 'g', t, y2, 'r', t, ys, 'b', linewidth=2)
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.legend({'cos(t/3)','cos(t/4)','cos(t/3)+cos(t/4)'})
plt.title('My cosines')
plt.savefig('demo02.png')
plt.show()

