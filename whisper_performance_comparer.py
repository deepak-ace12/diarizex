import whisperx

# from dotenv import load_dotenv
import torchaudio
from fastapi import FastAPI, File, UploadFile
import time
import torch
import numpy as np
from pydub.utils import mediainfo
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

local_cache = {}

model_name_en = "WAV2VEC2_ASR_BASE_960H"
model_name_es = "VOXPOPULI_ASR_BASE_10K_ES"
pipeline_type = "torchaudio"
whisperx_model_size = "base"
whisperx_beam_size = 1


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


@app.post("/init")
async def transcribe(
    beam_size: int,
    temperature: float,
    no_speech_threshold: float,
    sample_rate: int,
    model_size: str = "large-v2",
):
    print("model_size", model_size)
    local_cache["sample_rate"] = sample_rate
    local_cache["model"] = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        asr_options={
            "beam_size": beam_size,
            "no_speech_threshold": no_speech_threshold,
            "temperatures": [temperature],
        },
    )
    return f"model loaded successfull {model_size}"


def get_word_level_confidence_score(transcript_result):
    transcription_confidence_score = []
    transcript_segments = transcript_result["word_segments"]
    for word in transcript_segments:
        transcription_confidence_score.append(word.get("score", 0))
    arr = np.array(transcription_confidence_score)
    percentage_above_90 = np.sum(arr >= 0.9) / len(arr) * 100
    percentage_above_80 = np.sum(arr >= 0.8) / len(arr) * 100
    percentage_above_70 = np.sum(arr >= 0.7) / len(arr) * 100
    percentage_above_60 = np.sum(arr >= 0.6) / len(arr) * 100
    percentage_above_50 = np.sum(arr >= 0.5) / len(arr) * 100
    percentage_below_50 = np.sum(arr < 0.5) / len(arr) * 100
    median = np.median(arr)
    return {
        "score_above_0.9 (%)": round(percentage_above_90, 2),
        "score_above_0.8 (%)": round(percentage_above_80, 2),
        "score_above_0.7 (%)": round(percentage_above_70, 2),
        "score_above_0.6 (%)": round(percentage_above_60, 2),
        "score_above_0.5 (%)": round(percentage_above_50, 2),
        "score_below_0.5 (%)": round(percentage_below_50, 2),
        "median": round(median, 2),
    }


@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...), unique_key: str = None):
    model = local_cache["model"]
    start_time = time.time()
    audio_path = f"{audio_file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())
    audio_info = mediainfo(audio_path)
    audio_format = audio_info.get("format_name")
    sample_rate = int(audio_info.get("sample_rate", 0))
    if sample_rate != local_cache.get("sample_rate"):
        print("Sample Rate Not Equel")
        _audio = AudioSegment.from_file(audio_path)
        new_sample_rate = local_cache.get("sample_rate")
        audio = _audio.set_frame_rate(new_sample_rate)
        audio.export(audio_path, format=audio_format)
        sample_rate = local_cache.get("sample_rate")

    audio_duration = float(audio_info.get("duration", 0))
    audio = whisperx.load_audio(audio_path)
    print(f"Sample rate: {sample_rate} Hz")

    try:
        transcript = model.transcribe(audio, batch_size=batch_size)
        if transcript["language"] == "en":
            result = whisperx.align(
                transcript["segments"],
                align_model_en,
                align_metadata_en,
                audio,
                device,
                return_char_alignments=False,
            )
        elif transcript["language"] == "es":
            result = whisperx.align(
                transcript["segments"],
                align_model_es,
                align_metadata_es,
                audio,
                device,
                return_char_alignments=False,
            )
        else:
            model_a, metadata = whisperx.load_align_model(
                language_code=transcript["language"], device=device
            )
            result = whisperx.align(
                transcript["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
        end_time = time.time()
        total_time = end_time - start_time
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        scores = get_word_level_confidence_score(result)
        output = {
            "device": device,
            "batch_size": batch_size,
            "compute_type": compute_type,
            "file_unique_key": unique_key,
            "sample_rate": sample_rate,
            "audio_format": audio_format,
            "audio_duration": audio_duration,
            "transcription_time": total_time,
        }
        output.update(scores)
        output["transcript"] = result
        return " ".join([segment.get("text") for segment in result.get("segments")])
    except Exception as ex:
        import traceback

        traceback.print_exc()
        return {"error": str(ex)}
