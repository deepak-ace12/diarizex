import dask
import whisperx
import time

import nemo.collections.asr as nemo_asr
from pydub import AudioSegment

from pydub.silence import split_on_silence

silence_threshold = -40

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
    "nvidia/speakerverification_en_titanet_large"
)


def is_identical_speaker(audio_1, audio_2):
    return speaker_model.verify_speakers(audio_1, audio_2)


def cut_wav(input_wav, start_time, end_time, output_wav):
    audio = AudioSegment.from_wav(input_wav)
    cut_audio = audio[start_time:end_time]
    cut_audio.export(output_wav, format="wav")


device = "cpu"
batch_size = 16  # reduce if low on GPU mem
compute_type = "int8"

# device = "cuda"
# audio_file = "/Users/I1597/Downloads/5sec.wav"
# batch_size = 32 # reduce if low on GPU mem
# compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model(
    "base", device, compute_type=compute_type, asr_options={"beam_size": 1}
)

t1 = time.time()


def transcribe(audio_file, diarize=False):
    audio = whisperx.load_audio(audio_file)

    def diarize_audio():
        diarize_model = whisperx.DiarizationPipeline(
            device=device, use_auth_token="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"
        )
        diarize_segments = diarize_model(audio)
        print("Diarization", diarize_segments)
        return diarize_segments

    def transcribe_audio():
        transcript = model.transcribe(audio, batch_size=batch_size, task="translate")
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
        print("SEGMENTATION", result)
        return result

    def assign_word_speakers(diarize_segments, transcribed_segments):
        print("TRANSCRIBED SEGMENTS", transcribed_segments)
        print("DIARIZATION SEGMENTS", diarize_segments)
        word_segments = transcribed_segments.get("word_segments")
        abc = []
        index = 0
        for segment in diarize_segments.itertuples(index=False):
            sentence = ""
            for word in word_segments[index:]:
                index += 1
                print("Index", index)
                if word.get("word"):
                    if not word.get("start") or not word.get("end"):
                        sentence += " " + word.get("word")
                        continue
                    if (
                        word.get("start") >= segment.start
                        and word.get("end") <= segment.end
                    ):
                        sentence += " " + word.get("word")
                    else:
                        index -= 1
                        break
            if sentence:
                temp = {}
                temp["speaker"] = segment.speaker
                temp["start"] = round(segment.start, 3)
                temp["end"] = round(segment.end, 3)
                temp["text"] = sentence
                print("temp", temp)
                abc.append(temp)
        return abc

    try:
        if diarize:
            delay_diarize = dask.delayed(diarize_audio)()
            delay_transcribe = dask.delayed(transcribe_audio)()
            diarize_segments, transcribed_segments = dask.compute(
                delay_diarize, delay_transcribe
            )
            speaker_segments = assign_word_speakers(
                diarize_segments, transcribed_segments
            )
            # speaker_segments = dialogues["segments"]
            # for segment in speaker_segments:
            #     segment.pop("words")
            #     # print(f"{segment.get('speaker')}({int(segment.get('start')//60)}:{round(segment.get('start')%60, 2)} - {int(segment.get('end')//60)}:{round(segment.get('end')%60, 2)}): {segment.get('text')}")
            t2 = time.time()
            print("Total Time", (t2 - t1))
            print("Transcript", speaker_segments)
            return speaker_segments
        else:
            transcribed_segments = transcribe_audio()
            transcript = " ".join(
                [
                    segment.get("text")
                    for segment in transcribed_segments.get("segments", [])
                ]
            )
            print("Transcript", transcript)
            return transcript
    except Exception as ex:
        return {"error": str(ex)}


existing_speakers = []
unique_speakers = set()
audio_files = []
for i in range(33):
    audio_files.append(f"/Users/I1597/Downloads/mamoon_sandeep/{i}.wav")

# audio_files = ["/Users/I1597/Downloads/amit_v/20.wav"]
print(audio_files)
final_transcript = []
for audio_file in audio_files:

    """
    Start: Trying remove blanks from audio
    """
    # split_on_silence(
    #     audio_segment,
    #     min_silence_len=1000,
    #     silence_thresh=-16,
    #     keep_silence=100,
    #     seek_step=1,
    # )
    # s1 = time.time()
    # audio_ = AudioSegment.from_wav(audio_file)
    # audio_segments = split_on_silence(audio_, silence_thresh=silence_threshold, seek_step=500, min_silence_len=500, keep_silence=200)
    # merged_audio = AudioSegment.empty()

    # for seg in audio_segments:
    #     merged_audio += seg

    # merged_audio.export(audio_file, format="wav")
    # s2 = time.time()
    # print("Silence remove time", (s2-s1))
    """
    End: Trying remove blanks from audio
    """

    print("Audio File", audio_file)
    started = time.time()
    transcript_segments = transcribe(audio_file, True)
    replace_speaker = {}
    combined_speaker_data = {}
    for idx, segment in enumerate(transcript_segments):
        if type(segment) == str or not (segment.get("speaker")):
            print("Segment", segment)
            continue
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

    print("Combined Speaker", combined_speaker_data)
    for speaker_label, metadata in combined_speaker_data.items():
        output_file = "temp_" + speaker_label + str(int(time.time())) + ".wav"
        start_time = int(metadata.get("start") * 1000)
        end_time = int(metadata.get("end") * 1000)
        cut_wav(audio_file, start_time, end_time, output_file)
        if len(existing_speakers):
            match_found = False
            for speaker in existing_speakers:
                if is_identical_speaker(output_file, speaker.get("speaker_voice")):
                    print(
                        f"Current Chunk's {segment.get('speaker')} is same as previous chunk's {speaker.get('speaker_name')}"
                    )
                    print("Identical ", output_file, speaker.get("speaker_voice"))
                    if speaker_label not in replace_speaker:
                        replace_speaker[speaker_label] = speaker.get("speaker_name")
                        match_found = True
                        break
                else:
                    print("non identical", output_file, speaker.get("speaker_voice"))
            else:
                if not match_found:
                    speaker_name = f"speaker__{len(unique_speakers)}"
                    print("Match Not found block", speaker_name)
                    existing_speakers.append(
                        {
                            "speaker_id": len(existing_speakers) + 1,
                            "speaker_name": speaker_name,
                            "speaker_voice": output_file,
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
                    "speaker_voice": output_file,
                }
            )
            print("Unique Speakers ELSE BLOCK", unique_speakers)
            replace_speaker[speaker_label] = speaker_name
            unique_speakers.add(speaker_name)

    #         if len(existing_speakers):
    #             if not(segment.get("speaker")):
    #                 continue
    #             print("**********", segment.get("speaker"))
    #             output_file = "temp_" + segment.get("speaker") + str(int(time.time())) + ".wav"
    #             start_time = int(segment.get("start") * 1000)
    #             end_time = (segment.get("end") * 1000)
    #             print (audio_file, start_time, end_time)
    #             cut_wav(audio_file, start_time, end_time, output_file)
    #             match_found = False
    #             for speaker in existing_speakers:
    #                 if (is_identical_speaker(output_file, speaker.get("speaker_voice"))):
    #                     print(f"Current Chunk's {segment.get('speaker')} is same as previous chunk's {speaker.get('speaker_name')}")
    #                     print("Identical ", output_file, speaker.get("speaker_voice"))
    #                     if segment["speaker"] not in replace_speaker:
    #                         replace_speaker[segment["speaker"]] = speaker.get("speaker_name")
    #                     # segment["speaker"] = speaker.get("speaker_name")
    #                     match_found = True
    #                     break
    #                 else:
    #                     print("non identical", output_file, speaker.get("speaker_voice"))
    #             else:
    #                 if not match_found:
    #                     speaker_name = f"speaker__{len(unique_speakers)}"
    #                     print("Match block", speaker_name)
    #                     existing_speakers.append(
    #                         {
    #                             "speaker_id": len(existing_speakers) + 1,
    #                             "speaker_name": speaker_name,
    #                             "speaker_voice": output_file
    #                         }
    #                     )
    #                     print("Unique Speakers", unique_speakers)
    #                     segment["speaker"] = speaker_name
    #                     unique_speakers.add(speaker_name)

    #         else:
    #             temp = {}
    #             output_file = segment.get("speaker") + ".wav"
    #             start_time = int(segment.get("start") * 1000)
    #             end_time = int(segment.get("end") * 1000)
    #             print (audio_file, start_time, end_time)
    #             cut_wav(audio_file, start_time, end_time, output_file)
    #             speaker_name = f"speaker__{len(unique_speakers)}"
    #             print("Else block", speaker_name)
    #             existing_speakers.append(
    #                 {
    #                     "speaker_id": len(existing_speakers) + 1,
    #                     "speaker_name": speaker_name,
    #                     "speaker_voice": output_file

    #                 }
    #             )
    #             segment["speaker"] = speaker_name
    #             unique_speakers.add(speaker_name)

    if replace_speaker:
        for script in transcript_segments:
            script["speaker"] = replace_speaker.get(script.get("speaker"))

    print(transcript_segments)
    ended = time.time()
    print("Total Time", (ended - started))
    print("*" * 100)
    print()
    final_transcript.append(transcript_segments)


print("##" * 50)
print(final_transcript)


"""

from pydub import AudioSegment
audio = AudioSegment.from_wav("amit_v.wav")
start = 1
start = 100
for i in range(103):
   start = start - 100
   print("start", start)
   end = start + 5100
   cut_audio = audio[start:end]
   start = end
   output = f"amit_v/{i}.wav"
   cut_audio.export(output, format="wav")


"""
