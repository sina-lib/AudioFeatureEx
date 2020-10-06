from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

def STFT_matrix(inArr, windowLen, PAD_N=0):
    Y = []
    # if the last window is not consistent, pad with PAD_N
    if len(inArr) % windowLen != 0:
        x = windowLen - (len(inArr) % windowLen)
        inArr = np.append(inArr, [ PAD_N for i in range(x) ])
    # number of windows:
    N = int(len(inArr) / windowLen)
    for i in range(N):
        windowFourier = fft(inArr[i*windowLen:((i+1)*windowLen)])
        Y.append(windowFourier)
    return Y

if __name__ == '__main__':
    # create a simple signal
    mySignal_len = 2547
    myWindowLen = 120
    t = np.linspace(0,2.5,mySignal_len)
    y = np.sin(44*t) + 5 * np.sin(244 * t) + 0.2
    noise = 3 * np.random.rand(mySignal_len)

    Y = STFT_matrix(y,myWindowLen)
    # we the other half of frequencies is extra, omitting them:
    Real = [ abs(r)[0:int(myWindowLen/2)] for r in Y]
    plt.matshow(Real)
    plt.xlabel('Frequency - 0 to pi')
    plt.ylabel('Time')
    plt.show()


