import dask
import whisperx
import os

# from dotenv import load_dotenv
import torchaudio
from fastapi import FastAPI, File, UploadFile
import time
import torch
from typing import Optional


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


diarize_model = whisperx.DiarizationPipeline(device=device, use_auth_token="")


@app.post("/transcribe")
async def transcribe(
    audio_file: UploadFile = File(...),
    unique_key: Optional[str] = None,
    diarize: Optional[bool] = False,
    threshold: float = 0.7,
):
    start_time = time.time()
    if local_cache.get(unique_key):
        pass
    else:
        local_cache[unique_key] = {}
        local_cache[unique_key]["existing_speakers"] = []
        local_cache[unique_key]["unique_speakers"] = set()

    ct1 = time.time()
    flac_audio_path = f"{audio_file.filename}"
    with open(flac_audio_path, "wb") as f:
        f.write(await audio_file.read())
    ct11 = time.time()
    print("Reading file", (ct11 - ct1))
    audio_path = "temp_audio.wav"
    compressed_audio = AudioSegment.from_file(flac_audio_path, format="flac")
    compressed_audio.export(audio_path, format="wav")
    ct2 = time.time()
    print("converting file", ct2 - ct11)
    audio = whisperx.load_audio(audio_path)
    ct3 = time.time()
    print("Loading file", (ct3 - ct2))

    def diarize_audio():
        t2 = time.time()
        diarize_segments = diarize_model(audio)
        t3 = time.time()
        print("Diarization", (t3 - t2))
        return diarize_segments

    def transcribe_audio():
        t11 = time.time()
        transcript = model.transcribe(audio, batch_size=batch_size)
        t22 = time.time()
        print("Transcription time", (t22 - t11))
        # TODO: in whisperx.asr, add task=translate only when the language is non english
        # TODO: in whisperx.diarize, updated config.yaml path in Pipeline
        t1 = time.time()
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
        t2 = time.time()
        print("Alignment time", t2 - t1)
        return result

    def adjust_speaker_labels(speaker_segments):
        replace_speaker = {}
        combined_speaker_data = {}
        for idx, segment in enumerate(speaker_segments):
            if type(segment) == str or not (segment.get("speaker")):
                print("Segment", segment)
                continue

            # Find the longest consecutive segments of a speaker
            if idx == 0:
                current_speaker = segment.get("speaker")
                current_start = segment.get("start")
                current_end = segment.get("end")
            elif current_speaker == segment.get("speaker"):
                current_end = segment.get("end")
            else:
                current_speaker = segment.get("speaker")
                current_start = segment.get("start")
                current_end = segment.get("end")
            # replace the saved segment if the current segment length is longer than the saved one
            if current_speaker and current_speaker in combined_speaker_data:
                if (current_end - current_start) > (
                    combined_speaker_data.get(current_speaker, {}).get("end", 0)
                    - combined_speaker_data.get(current_speaker, {}).get("start", 0)
                ):
                    combined_speaker_data[current_speaker] = {
                        "start": current_start,
                        "end": current_end,
                    }
            else:
                combined_speaker_data[current_speaker] = {
                    "start": current_start,
                    "end": current_end,
                }

        for speaker_label, metadata in combined_speaker_data.items():
            output_file = "temp_" + speaker_label + str(int(time.time())) + ".wav"
            start_time = int(metadata.get("start") * 1000)
            end_time = int(metadata.get("end") * 1000)
            cut_wav(audio_path, start_time, end_time, output_file)
            existing_speakers = local_cache.get(unique_key, {}).get("existing_speakers")
            unique_speakers = local_cache.get(unique_key, {}).get("unique_speakers")
            if len(existing_speakers):
                match_found = False
                for speaker in existing_speakers:
                    if is_identical_speaker(
                        output_file, speaker.get("speaker_audio_file"), threshold
                    ):
                        if speaker_label not in replace_speaker:
                            replace_speaker[speaker_label] = speaker.get("speaker_name")
                            match_found = True
                            os.remove(output_file)
                            break
                    else:
                        print(
                            "non identical",
                            output_file,
                            speaker.get("speaker_audio_file"),
                        )
                else:
                    if not match_found:
                        speaker_name = f"speaker__{len(unique_speakers)}"
                        print("Match Not found block", speaker_name)
                        existing_speakers.append(
                            {
                                "speaker_id": len(existing_speakers) + 1,
                                "speaker_name": speaker_name,
                                "speaker_audio_file": output_file,
                            }
                        )
                        print("Unique Speakers", unique_speakers)
                        replace_speaker[speaker_label] = speaker_name
                        unique_speakers.add(speaker_name)
            else:
                speaker_name = f"speaker__{len(unique_speakers)}"
                existing_speakers.append(
                    {
                        "speaker_id": len(existing_speakers) + 1,
                        "speaker_name": speaker_name,
                        "speaker_audio_file": output_file,
                    }
                )
                print("Unique Speakers ELSE BLOCK", unique_speakers)
                replace_speaker[speaker_label] = speaker_name
                unique_speakers.add(speaker_name)

        if replace_speaker:
            for script in speaker_segments:
                script.pop("words", "")
                script["speaker"] = replace_speaker.get(script.get("speaker"))
        print("speaker_segments", speaker_segments)
        return speaker_segments

    try:
        if diarize:
            delay_diarize = dask.delayed(diarize_audio)()
            delay_transcribe = dask.delayed(transcribe_audio)()
            diarize_segments, transcribed_segments = dask.compute(
                delay_diarize, delay_transcribe
            )
            t11 = time.time()
            dialogues = whisperx.assign_word_speakers(
                diarize_segments, transcribed_segments
            )
            t22 = time.time()
            print("Assign word time", (t22 - t11))
            speaker_segments = dialogues["segments"]
            t1 = time.time()
            result = adjust_speaker_labels(speaker_segments)
            t2 = time.time()
            print("Label Adjust Time", (t2 - t1))
            os.remove(audio_path)
            t3 = time.time()
            print("Deleting file", (t3 - t2))
            end_time = time.time()
            print(
                "######################################### total request time ",
                (end_time - start_time),
            )
            return result
        else:
            transcribed_segments = transcribe_audio()
            transcript = " ".join(
                [
                    segment.get("text")
                    for segment in transcribed_segments.get("segments", [])
                ]
            )
            os.remove(audio_path)
            return transcript
    except Exception as ex:
        import traceback

        traceback.print_exc()
        return {"error": str(ex)}
