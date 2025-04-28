import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


class BitGenerator:
    def __init__(self, n_bits):
        self.n_bits = n_bits

    def generate(self):
        return np.random.choice([1, 0], self.n_bits)

class PulseShaper:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.pulse_one = np.ones(self.n_samples) 
        self.pulse_zero = -1 * self.pulse_one

    def shape(self, bits):
        
        time = []
        signal = []
        
        for i, bit in enumerate(bits):
            time.extend(np.linspace(i, i + 1, self.n_samples))
            signal.extend(self.pulse_one if bit == 1 else self.pulse_zero)

        return np.array(time), np.array(signal)

class AWGNChannel:
    def __init__(self, snr_db):
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)

    def transmit(self, signal, samples_per_bit):
        
        normalized_signal = signal / np.sqrt(samples_per_bit)
        noise_std = np.sqrt(1 / (2 * self.snr_linear))

        # Generate noise and add it
        noise = np.random.normal(0, noise_std, size=signal.shape)
        return normalized_signal + noise


class MatchedFilter:
    def __init__(self, filter_pulse, n_samples):
        self.filter = np.copy(filter_pulse[::-1])
        self.n_samples = n_samples


    def apply(self, signal):
        output = convolve(signal, self.filter, mode='same')
        return output

class IdentityFilter:
    def __init__(self, n_samples):
        self.filter = np.ones(n_samples)

    def apply(self, signal):
        return signal  # Pass the signal as it is

class RampFilter:
    def __init__(self, n_samples):
        self.filter = np.sqrt(3) * np.arange(n_samples) / n_samples
        self.n_samples = n_samples

    def apply(self, signal):
        output = convolve(signal, self.filter, mode='same')
        return output
    
class Sampler:
    def __init__(self, n_samples, n_bits, position='center'):
        self.n_samples = n_samples
        self.n_bits = n_bits
        self.position = position  # 'center' or 'last'

    def sample(self, signal):
        if self.position == 'center':
            indices = [self.n_samples * i + self.n_samples // 2 for i in range(self.n_bits)]
        elif self.position == 'last':
            indices = [self.n_samples * (i + 1) - 1 for i in range(self.n_bits)]
        else:
            raise ValueError("Unsupported position! Use 'center' or 'last'.")
        return np.array([signal[idx] for idx in indices])

