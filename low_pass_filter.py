import numpy as np
from scipy import signal
import matplotlib.pyplot as plt



if __name__ == "__main__":
    t = np.linspace(0, 1.0, 2001)
    xlow = np.sin(2 * np.pi * 5 * t)
    xhigh = np.sin(2 * np.pi * 250 * t)
    x = xlow + xhigh

    b, a = signal.butter(8, 0.125)
    y = signal.filtfilt(b, a, x, padlen=150)
    print(np.abs(y - xlow).max())

    plt.figure()
    plt.plot(t, x, marker="o", linestyle="-")
    plt.plot(t, y, marker="x", linestyle="-")
    plt.xlabel("Time")
    plt.ylabel("signal")
    plt.title(f"Signal vs. time")
    plt.show()

    print(y)