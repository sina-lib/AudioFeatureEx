from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,2.5,2500)
y = np.sin(44*t) + 5 * np.sin(244 * t) + 0.2
noise = np.random.rand(2500)
yf = fft(y+noise)

plt.subplot(211)
plt.plot(t,y)
plt.subplot(212)
plt.plot(t,yf)

plt.show()

