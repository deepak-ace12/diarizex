import dask
import whisperx
import os
from scipy.io import wavfile

# from dotenv import load_dotenv
import torchaudio
from fastapi import FastAPI, File, UploadFile, Query, Request
import time
import torch
from typing import Optional, List
import numpy as np
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment

# load_dotenv()

app = FastAPI()

if torch.cuda.is_available():
    device = "cuda"
    batch_size = 32
    compute_type = "float16"  #
else:
    device = "cpu"
    batch_size = 16
    compute_type = "int8"


model_name_en = "WAV2VEC2_ASR_BASE_960H"
model_name_es = "VOXPOPULI_ASR_BASE_10K_ES"

pipeline_type = "torchaudio"

bundle_en = torchaudio.pipelines.__dict__[model_name_en]
align_model_en = bundle_en.get_model().to(device)
labels_en = bundle_en.get_labels()
align_dictionary_en = {c.lower(): i for i, c in enumerate(labels_en)}
align_metadata_en = {
    "language": "en",
    "dictionary": align_dictionary_en,
    "type": pipeline_type,
}


bundle_es = torchaudio.pipelines.__dict__[model_name_es]
align_model_es = bundle_es.get_model().to(device)
labels_es = bundle_es.get_labels()
align_dictionary_es = {c.lower(): i for i, c in enumerate(labels_es)}
align_metadata_es = {
    "language": "es",
    "dictionary": align_dictionary_es,
    "type": pipeline_type,
}

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
    "nvidia/speakerverification_en_titanet_large"
)


def is_identical_speaker(audio_1, audio_2, threshold):
    return speaker_model.verify_speakers(audio_1, audio_2, threshold)


def cut_wav(input_wav, start_time, end_time, output_wav):
    audio = AudioSegment.from_wav(input_wav)
    cut_audio = audio[start_time:end_time]
    cut_audio.export(output_wav, format="wav")


model = whisperx.load_model(
    "base", device, compute_type=compute_type, asr_options={"beam_size": 1}
)

local_cache = {}


@app.get("/end_transcription")
async def get_request(unique_key: str):
    start_time = time.time()
    existing_speakers = local_cache.pop(unique_key, {}).get("existing_speakers")
    print(existing_speakers)
    if existing_speakers:
        for speaker in existing_speakers:
            print("PATH EXISTS", os.path.exists(speaker.get("speaker_audio_file")))
            if os.path.exists(speaker.get("speaker_audio_file")):
                os.remove(speaker.get("speaker_audio_file"))
    # local_cache.pop(unique_key, None)
    return {"message": f"data for {unique_key} deleted from local cache"}


diarize_model = whisperx.DiarizationPipeline(
    device=device, use_auth_token="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"
)

from pydantic import BaseModel


class NumpyArrayRequest(BaseModel):
    data: list  # This should match the JSON-compatible format of your NumPy array


@app.post("/transcribe")
async def transcribe(
    data: NumpyArrayRequest,
    unique_key: Optional[str] = None,
    diarize: Optional[bool] = False,
    threshold: float = 0.7,
):
    audio_nparray = np.array(data.data, dtype=np.float32)
    diarize_segments = diarize_model(audio_nparray)
    transcript = model.transcribe(audio_nparray, batch_size=batch_size)
    print("Diarization", diarize_segments)
    output_filename = "output_demo.wav"
    t1 = time.time()
    wavfile.write(output_filename, 16000, audio_nparray)
    t2 = time.time()
    print("Conversion time", (t2 - t1))
    return transcript
