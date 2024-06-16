import numpy as np
import queue
import threading
import pyaudio
import matplotlib.pyplot as plt
from scipy import fftpack

# キュー
data_queue = queue.Queue()

def record_thread(index, samplerate, frames_per_buffer):
    """リアルタイムに音声を録音するスレッド"""

    # PyAudioインスタンスの生成とストリーム開始
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=samplerate,
                     input=True, input_device_index=index, frames_per_buffer=frames_per_buffer)
    # リアルタイム録音
    try:
        while True:
            data = stream.read(frames_per_buffer, exception_on_overflow=False)
            data = np.frombuffer(data, dtype="int16") / float((np.power(2, 16) / 2) - 1)
            data_queue.put(data)
            print(len(data_queue.queue))
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

def calc_fft(data, samplerate):
    """FFTを計算する関数"""

    spectrum = fftpack.fft(data)
    amp = np.sqrt((spectrum.real ** 2) + (spectrum.imag ** 2))
    amp = amp / (len(data) / 2)
    phase = np.arctan2(spectrum.imag, spectrum.real)
    phase = np.degrees(phase)
    freq = np.linspace(0, samplerate, len(data))

    return spectrum, amp, phase, freq

def plot_waveform(samplerate):
    """波形をプロットする関数"""

    # プロットの設定
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Amplitude')

    ax1.set_ylim(-1, 1)
    ax2.set_yscale('log')
    ax2.set_xlim(0, 10000)
    ax2.set_ylim(0.00001, 1)

    line1, = ax1.plot([], [], label='Time waveform', lw=1, color='red')
    line2, = ax2.plot([], [], label='Amplitude', lw=1, color='blue')

    while plt.fignum_exists(fig.number):
        if not data_queue.empty():
            data = data_queue.get()
            time_axis = np.linspace(0, len(data) / samplerate, num=len(data))
            spectrum, amp, phase, freq = calc_fft(data, samplerate)

            line1.set_data(time_axis, data)
            line2.set_data(freq[:len(freq) // 2], amp[:len(amp) // 2])

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()

            fig.tight_layout()
            try:
                plt.pause(0.01)
            except Exception as e:
                print('Error')

if __name__ == '__main__':
    """メイン文"""

    # 録音設定：サンプリングレート/フレームサイズ/マイクチャンネル
    samplerate = 44100
    frames_per_buffer = 4096
    index = 0

    # 測定スレッド
    threading.Thread(target=record_thread, args=(index, samplerate, frames_per_buffer), daemon=True).start()

    # プロットスレッド
    plot_waveform(samplerate)