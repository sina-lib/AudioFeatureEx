from scipy.fftpack import fft, ifft
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

def Cepstrum_matrix(inArr, windowLen, PAD_N=0):
    Y = []
    # if the last window is not consistent, pad with PAD_N
    if len(inArr) % windowLen != 0:
        x = windowLen - (len(inArr) % windowLen)
        inArr = np.append(inArr, [ PAD_N for i in range(x) ])
    # number of windows:
    N = int(len(inArr) / windowLen)
    for i in range(N):
        windowFourier = fft(inArr[i*windowLen:((i+1)*windowLen)])
        windowFourier = np.abs(windowFourier) ** 2
        windowFourier = np.where(windowFourier == 0, np.finfo(float).eps, windowFourier)
        windowFourier = np.log(windowFourier)
        cps = np.abs(ifft(windowFourier))
        Y.append(cps ** 2)
    return Y


if __name__ == '__main__':
    # create a simple signal
    mySignal_len = 6000
    myWindowLen = 25
    t = np.linspace(0,6,mySignal_len)
    y = 25 * np.sin(2 * np.pi * 70*t) + 150 * np.sin(2 * np.pi * 370*t)  + 50 * np.sin(2 * np.pi * 490 * t) + 45
    noise = 20 * np.random.rand(mySignal_len)
    y = y + noise

    Y = STFT_matrix(y,myWindowLen)
    Y = np.transpose(Y)
    # we the other half of frequencies is extra, omitting them:
    Real = [ abs(r)[0:int(myWindowLen/2)] for r in Y]
    plt.matshow(Real)
    plt.ylabel('Frequency - 0 to pi')
    plt.xlabel('Time')

    CEPS = Cepstrum_matrix(y,myWindowLen)

    CEPS = np.transpose(CEPS)
    CEPS = [ np.abs(i) for i in CEPS]
    plt.matshow(CEPS)

    plt.show()


