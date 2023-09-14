import numpy as np
import ffmpeg
from scipy.io import wavfile

# def audio_to_numpy(audio_file_path):
#     sample_rate, audio_data = wavfile.read(audio_file_path)
#     return sample_rate, audio_data


def audio_to_numpy(file: str, sr: int = 16000):
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    # return np.frombuffer(out, np.int16).flatten().astype(np.float16) / 16384.0
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def numpy_to_wav(np_array, sample_rate=16000):
    output_filename = "output.wav"
    wavfile.write(output_filename, sample_rate, np_array)
