import json


unique_key = "CAR0001"
fillers = ["um,", "Um,", "um", "Um", "Uh,", "uh,", "Uh", "uh"]


def get_original_contents_as_json(original_script_filepath):
    contents = []
    with open(original_script_filepath, "r") as file:
        file_contents = file.read()
        for line in file_contents.splitlines():
            if line:
                try:
                    speaker, text = line.split(":")
                except Exception:
                    speaker, text = line.split(";")
                text = (
                    text.replace("uh,", "")
                        .replace("Uh,", "")
                        .replace("uh", "")
                        .replace("Uh", "")
                        .replace("um,", "")
                        .replace("Um,", "")
                        .replace("um", "")
                        .replace("Um", "")
                )
                
                contents.append({"speaker": speaker.strip(), "text": text.strip()})
    return contents


def get_transcripted_contents_chunks(transcript_file_path):
    speaker_segments = json.loads(transcript_file_path)
    updated_segments = []
    for chunk in speaker_segments:
        for segment in chunk.get("segments"):
            speaker_label = "Agent" if segment.get("speaker") == "speaker__0" else "Patient"
            if len(updated_segments) > 0:
                if speaker_label == updated_segments[-1].get("speaker"):
                    updated_segments[-1]["text"] += " " + segment.get("text")
                    continue
            updated_segments.append({
                "speaker": speaker_label,
                "text": segment.get("text")
            })
    return updated_segments