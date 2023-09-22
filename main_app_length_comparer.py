import dask
import whisperx
import os

# from dotenv import load_dotenv
import torchaudio
from fastapi import FastAPI, File, UploadFile
import time
import torch
from typing import Optional
import numpy as np

import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
from pydub.utils import mediainfo

# load_dotenv()
max_chunk_size = 15
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
nemo_speaker_model = "nvidia/speakerverification_en_titanet_large"
whisperx_model_size = "base"
whisperx_beam_size = 1
max_comparable_audio_length = 2300
min_comparable_audio_length = 500
min_similarity_threshold = 0.55

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
    nemo_speaker_model
)


def is_identical_speaker(audio_1, audio_2, threshold):
    return speaker_model.verify_speakers(audio_1, audio_2, threshold)


# def cut_wav(input_wav, start_time, end_time, output_wav):
#     audio = AudioSegment.from_wav(input_wav)
#     cut_audio = audio[start_time:end_time]
#     cut_audio.export(output_wav, format="wav")


def cut_audio(input_file, start_time, end_time, output_file, audio_format):
    audio = AudioSegment.from_file(input_file, format=audio_format)
    cut_audio = audio[start_time:end_time]
    cut_audio.export(output_file, format=audio_format)


model = whisperx.load_model(
    whisperx_model_size,
    device,
    compute_type=compute_type,
    asr_options={"beam_size": whisperx_beam_size},
)

local_cache = {}


@app.get("/end_transcription")
async def get_request(unique_key: str):
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


def get_word_level_confidence_score(transcript_result):
    transcription_confidence_score = []
    transcript_segments = transcript_result["word_segments"]
    for word in transcript_segments:
        transcription_confidence_score.append(word.get("score", 0))
    arr = np.array(transcription_confidence_score)
    percentage_above_0_9 = np.sum(arr >= 0.9) / len(arr) * 100
    percentage_above_0_8 = np.sum(arr >= 0.8) / len(arr) * 100
    percentage_above_0_7 = np.sum(arr >= 0.7) / len(arr) * 100
    percentage_above_0_6 = np.sum(arr >= 0.6) / len(arr) * 100
    percentage_above_0_5 = np.sum(arr >= 0.5) / len(arr) * 100
    percentage_below_0_5 = np.sum(arr < 0.5) / len(arr) * 100
    average = np.mean(arr)
    return {
        "percentage_above_0_9": round(percentage_above_0_9, 2),
        "percentage_above_0_8": round(percentage_above_0_8, 2),
        "percentage_above_0_7": round(percentage_above_0_7, 2),
        "percentage_above_0_6": round(percentage_above_0_6, 2),
        "percentage_above_0_5": round(percentage_above_0_5, 2),
        "percentage_below_0_5": round(percentage_below_0_5, 2),
        "average": round(average, 2),
    }


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        transcription_confidence_score = []
        # assign speaker to segment (if any)
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], seg["end"]
        ) - np.maximum(diarize_df["start"], seg["start"])
        diarize_df["union"] = np.maximum(diarize_df["end"], seg["end"]) - np.minimum(
            diarize_df["start"], seg["start"]
        )
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df["intersection"] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = (
                dia_tmp.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
            seg["speaker"] = speaker

        # assign speaker to words
        if "words" in seg:
            for word in seg["words"]:
                transcription_confidence_score.append(word.get("score", 0))
                if "start" in word:
                    diarize_df["intersection"] = np.minimum(
                        diarize_df["end"], word["end"]
                    ) - np.maximum(diarize_df["start"], word["start"])
                    diarize_df["union"] = np.maximum(
                        diarize_df["end"], word["end"]
                    ) - np.minimum(diarize_df["start"], word["start"])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df["intersection"] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = (
                            dia_tmp.groupby("speaker")["intersection"]
                            .sum()
                            .sort_values(ascending=False)
                            .index[0]
                        )
                        word["speaker"] = speaker
            arr = np.array(transcription_confidence_score)
            percentage_above_0_9 = np.sum(arr > 0.9) / len(arr) * 100
            percentage_above_0_8 = np.sum(arr > 0.8) / len(arr) * 100
            percentage_above_0_7 = np.sum(arr > 0.7) / len(arr) * 100
            percentage_below_0_7 = np.sum(arr < 0.7) / len(arr) * 100
            average = np.mean(arr)
            seg["scores"] = {
                "percentage_above_0_9": percentage_above_0_9,
                "percentage_above_0_8": percentage_above_0_8,
                "percentage_above_0_7": percentage_above_0_7,
                "percentage_below_0_7": percentage_below_0_7,
                "average": average,
            }
    return (transcript_result,)


@app.post("/transcribe")
async def transcribe(
    audio_file: UploadFile = File(...),
    unique_key: Optional[str] = None,
    diarize: Optional[bool] = False,
    max_num_speakers: Optional[bool] = 2,
):
    start_time = time.time()
    if local_cache.get(unique_key):
        pass
    else:
        local_cache[unique_key] = {}
        local_cache[unique_key]["existing_speakers"] = []
        local_cache[unique_key]["unique_speakers"] = set()

    # audio_path = f"{audio_file.filename}"
    # with open(audio_path, "wb") as f:
    #     f.write(await audio_file.read())

    audio_path = f"{audio_file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())
    audio_info = mediainfo(audio_path)
    sample_rate = int(audio_info.get("sample_rate", 0))
    audio_format = audio_info.get("format_name")
    audio_duration = float(audio_info.get("duration", 0))
    audio = whisperx.load_audio(audio_path)
    # from scipy.io import wavfile
    # ttt1 = time.time()
    # sample_rate, _ = wavfile.read(audio_path)  # Replace with the path to your WAV audio file
    print(f"Sample rate: {sample_rate} Hz")
    # ttt2 = time.time()
    # print("Sample Rate Time", (ttt2-ttt1))

    def diarize_audio():
        t2 = time.time()
        diarized_segments = diarize_model(audio)
        t3 = time.time()
        print("Diarization", (t3 - t2))
        print("Diarized Segments", diarized_segments)
        return diarized_segments

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
        print("Transcription", result)
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
                current_speaker = segment.get("speaker", "")
                current_start = segment.get("start", "")
                current_end = segment.get("end", "")
                current_text = segment.get("text", "")
            elif current_speaker == segment.get("speaker", ""):
                current_end = segment.get("end", "")
                current_text = current_text + " " + segment.get("text")
            else:
                current_speaker = segment.get("speaker", "")
                current_start = segment.get("start", "")
                current_end = segment.get("end", "")
                current_text = segment.get("text", "")
            # replace the saved segment if the current segment length is longer than the saved one
            if current_speaker and current_speaker in combined_speaker_data:
                if (current_end - current_start) > (
                    combined_speaker_data.get(current_speaker, {}).get("end", 0)
                    - combined_speaker_data.get(current_speaker, {}).get("start", 0)
                ):
                    combined_speaker_data[current_speaker] = {
                        "start": current_start,
                        "end": current_end,
                        "text": current_text,
                    }
            else:
                combined_speaker_data[current_speaker] = {
                    "start": current_start,
                    "end": current_end,
                    "text": current_text,
                }
        print("Combined Data", combined_speaker_data)
        max_similarity_score = 0
        best_matching_speaker = None
        set_threshold = 0
        for speaker_label, metadata in combined_speaker_data.items():
            output_file = (
                f"SAMPLE_{speaker_label}_{str(int(time.time()))}.{audio_format}"
            )
            start_time = int(metadata.get("start") * 1000)
            end_time = int(metadata.get("end") * 1000)
            audio_length = end_time - start_time
            # cut_wav(audio_path, start_time, end_time, output_file)
            cut_audio(audio_path, start_time, end_time, output_file, audio_format)
            existing_speakers = local_cache.get(unique_key, {}).get("existing_speakers")
            unique_speakers = local_cache.get(unique_key, {}).get("unique_speakers")
            if len(existing_speakers):
                match_found = False
                for speaker in existing_speakers:
                    print("Lengths", audio_length, speaker.get("audio_length"))
                    if (
                        audio_length >= max_comparable_audio_length
                        and speaker.get("audio_length") >= max_comparable_audio_length
                    ):
                        threshold = 0.75
                    elif (
                        audio_length >= min_comparable_audio_length * 4
                        and speaker.get("audio_length")
                        >= min_comparable_audio_length * 4
                    ):
                        threshold = 0.72
                    elif (
                        audio_length >= min_comparable_audio_length * 3
                        and speaker.get("audio_length")
                        >= min_comparable_audio_length * 3
                    ):
                        threshold = 0.7
                    elif (
                        audio_length >= min_comparable_audio_length * 2
                        and speaker.get("audio_length")
                        >= min_comparable_audio_length * 2
                    ):
                        threshold = 0.65
                    elif (
                        audio_length >= min_comparable_audio_length
                        and speaker.get("audio_length") >= min_comparable_audio_length
                    ):
                        threshold = 0.6
                    else:
                        threshold = 0.55
                    if sample_rate < 16000:
                        threshold -= 0.05
                    print("Threshold", threshold)
                    identical, similarity_score = is_identical_speaker(
                        output_file, speaker.get("speaker_audio_file"), threshold
                    )
                    if identical:
                        if speaker_label not in replace_speaker:
                            # replace_speaker[speaker_label] = speaker.get("speaker_name")
                            replace_speaker[speaker_label] = {
                                "speaker": speaker.get("speaker_name"),
                                "is_new_speaker": False,
                            }
                            print(
                                "MATCH FOUND",
                                metadata.get("text"),
                                similarity_score,
                                speaker_label,
                                speaker.get("speaker_name"),
                            )
                            match_found = True
                            os.remove(output_file)
                            break
                    else:
                        if similarity_score.item() > max_similarity_score:
                            print(
                                f"Most Similar Speaker {speaker.get('speaker_name')} prob: {similarity_score}"
                            )
                            max_similarity_score = round(similarity_score.item(), 2)
                            best_matching_speaker = speaker.get("speaker_name")
                            set_threshold = threshold
                        print(
                            "non identical",
                            metadata.get("text"),
                            output_file,
                            speaker.get("speaker_audio_file"),
                            speaker_label,
                            similarity_score,
                        )
                else:
                    if not match_found:
                        if len(existing_speakers) >= max_num_speakers and (
                            (
                                set_threshold >= 0.60
                                and set_threshold - 0.05 <= max_similarity_score
                            )
                            or (
                                audio_length <= 300
                                and round(max_similarity_score, 1) >= 0.5
                            )
                            or (
                                len(metadata.get("text", "").strip().split()) <= 2
                                and round(max_similarity_score, 2) >= 0.55
                            )
                        ):
                            print(
                                "Adjusting the threshold",
                                metadata.get("text"),
                                set_threshold,
                                max_similarity_score,
                                best_matching_speaker,
                            )
                            # replace_speaker[speaker_label] = best_matching_speaker
                            replace_speaker[speaker_label] = {
                                "speaker": best_matching_speaker,
                                "is_new_speaker": False,
                            }
                        elif (
                            audio_length < 500
                            or len(metadata.get("text", "").strip().split()) <= 2
                        ):
                            print(
                                "Audio too small to save for sampling",
                                metadata.get("text"),
                                set_threshold,
                                max_similarity_score,
                                best_matching_speaker,
                            )
                            replace_speaker[speaker_label] = {
                                "speaker": "unknown",
                                "is_new_speaker": True,
                                "best_matched_speaker": best_matching_speaker
                                if len(existing_speakers) >= 2
                                else "unknown",
                                "probability": max_similarity_score
                                if len(existing_speakers) >= 2
                                else 0,
                            }
                        else:
                            speaker_name = f"speaker__{len(unique_speakers)}"
                            print(
                                "Match Not found block",
                                metadata.get("text"),
                                speaker_name,
                                max_similarity_score,
                                best_matching_speaker,
                            )
                            existing_speakers.append(
                                {
                                    "speaker_id": len(existing_speakers) + 1,
                                    "speaker_name": speaker_name,
                                    "speaker_audio_file": output_file,
                                    "audio_length": audio_length,
                                }
                            )
                            print("Unique Speakers", unique_speakers)
                            # replace_speaker[speaker_label] = speaker_name
                            if len(existing_speakers) <= max_num_speakers:
                                replace_speaker[speaker_label] = {
                                    "speaker": speaker_name,
                                    "is_new_speaker": False,
                                }
                            else:
                                replace_speaker[speaker_label] = {
                                    "speaker": speaker_name,
                                    "is_new_speaker": True,
                                    "best_matched_speaker": best_matching_speaker,
                                    "probability": max_similarity_score,
                                }
                            unique_speakers.add(speaker_name)
            else:
                speaker_name = f"speaker__{len(unique_speakers)}"
                existing_speakers.append(
                    {
                        "speaker_id": len(existing_speakers) + 1,
                        "speaker_name": speaker_name,
                        "speaker_audio_file": output_file,
                        "audio_length": audio_length,
                    }
                )
                print("Unique Speakers ELSE BLOCK", unique_speakers)
                # replace_speaker[speaker_label] = speaker_name
                replace_speaker[speaker_label] = {
                    "speaker": speaker_name,
                    "is_new_speaker": False,
                }
                unique_speakers.add(speaker_name)

        # if replace_speaker:
        #     print("REPLACE SPEAKER 111111111", replace_speaker)
        #     for script in speaker_segments:
        #         script.pop("words", "")
        #         script["speaker"] = replace_speaker.get(script.get("speaker"))

        if replace_speaker:
            for script in speaker_segments:
                spkr = replace_speaker.get(script.get("speaker"))
                script.pop("words", "")
                script["speaker"] = spkr.get("speaker")
                if spkr.get("is_new_speaker"):
                    script["is_new_speaker"] = True
                    script["best_matched_speaker"] = spkr.get("best_matched_speaker")
                    script["probability"] = spkr.get("probability")

        print("speaker_segments", speaker_segments)
        return speaker_segments

    try:
        if diarize:
            delay_diarize = dask.delayed(diarize_audio)()
            delay_transcribe = dask.delayed(transcribe_audio)()
            diarized_segments, transcribed_segments = dask.compute(
                delay_diarize, delay_transcribe
            )
            t11 = time.time()
            dialogues = whisperx.assign_word_speakers(
                diarized_segments, transcribed_segments
            )
            t22 = time.time()
            print("Assign word time", (t22 - t11))
            speaker_segments = dialogues["segments"]
            print("Speaker_Segments", speaker_segments)
            t1 = time.time()
            if audio_duration < max_chunk_size:
                result = adjust_speaker_labels(speaker_segments)
            else:
                for script in speaker_segments:
                    script.pop("words", "")
                result = speaker_segments
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
