import pandas as pd
import time
import json
from fuzzywuzzy import fuzz
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pydub import AudioSegment
import string
import random
import requests

# t1 = time.time()

# t2 = time.time()
# print("total time", (t2-t1))
# TODO: Add your audio paths
audios = [
    "/Users/I1597/Downloads/Data/Audio Recordings/CAR0001.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/CAR0002.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/CAR0003.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GAS0003.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GAS0004.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GAS0007.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0001.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0003.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0004.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0001.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0003.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0127.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0128.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0042.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GEN0001.mp3",
]


stop_words = set(stopwords.words("english"))
nltk.download("punkt")
nltk.download("stopwords")

base_url = "https://ci-develop.innovaccer.net/transcription_backend/v1" # TODO : Add GPU URL

base_path_for_chunks = "/Users/I1597/Downloads" # TODO : Change Path
original_transcript_base_path = "/Users/I1597/Downloads/Data/original_transcription" # TODO : Change Path
permformance_output_base_path = "/Users/I1597/Downloads/chunk_performace" # TODO : Change Path


def write_chunks_to_readable_file(speaker_segments, output_file):
    updated_segments = []
    for chunk in speaker_segments:
        for segment in chunk.get("segments"):
            speaker_label = segment.get("speaker")
            if len(updated_segments) > 0:
                if speaker_label == updated_segments[-1].get("speaker"):
                    updated_segments[-1]["text"] += " " + segment.get("text").strip().replace("  ", " ")
                    continue
            updated_segments.append({
                "speaker": speaker_label,
                "text": segment.get("text").strip().replace("  ", " ")
            })
    template = ""
    for segment in updated_segments:
        template = template + segment.get("speaker") + ": " + segment.get("text") + "\n\n"
    
    with open(output_file, 'w') as file:
        file.write(template)


def get_jaccard_similarity(text1, text2):
    def tokenize_text(text):
        words = word_tokenize(text)
        # Remove stop words and convert to lowercase
        words = [
            word.lower()
            for word in words
            if word.isalnum() and word.lower() not in stop_words
        ]
        return set(words)

    word_set1 = tokenize_text(text1)
    word_set2 = tokenize_text(text2)
    intersection = len(word_set1.intersection(word_set2))
    union = len(word_set1.union(word_set2))
    jaccard_similarity = intersection / union

    return round(jaccard_similarity * 100, 2)


def get_cosine_similarity_tfidf(text1, text2):
    vectorizer = TfidfVectorizer()

    # Compute TF-IDF vectors for the two texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute cosine similarity between the two TF-IDF vectors
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return round(cosine_sim[0][0] * 100, 2)


def get_cosine_similarity_count(text1, text2):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(X[0], X[1])
    return round(cosine_sim[0][0]*100, 2)


def generate_random_string(length=8):
    digits = string.digits  # '0123456789'
    random_string = ''.join(random.choice(digits) for _ in range(length))
    return random_string


def get_original_transcript_txt(file_path):
    txt = ""

    with open(file_path, "r") as file:
        file_contents = file.read()
        for line in file_contents.splitlines():
            if line:
                txt += (
                    line.replace("P: ", " ")
                    .replace("D: ", " ")
                    .replace("uh,", "")
                    .replace("Uh,", "")
                    .replace("uh", "")
                    .replace("Uh", "")
                    .replace("um,", "")
                    .replace("Um,", "")
                    .replace("um", "")
                    .replace("Um", "")
                )
    txt = txt.replace("  ", " ")
    return txt


def get_chunk_transcription(request_id, audio_path, chunk_duration, audio_file_key):
    diarize = True # whether you want diarization default: False
    language = "en" # Optional parameter, if privided, transcription will be a little faster
    max_num_speakers = 2 # optional, if provided, diarization will have a little improvement
    min_num_speakers = 1 # optional, if provided, diarization will have a little improvement
    chunk_folder_path = f"{base_path_for_chunks}/{audio_file_key}/{chunk_duration}"
    audio_format = "mp3"
    audio = AudioSegment.from_file(audio_path, format=audio_format)
    if not os.path.exists(chunk_folder_path):
        os.makedirs(chunk_folder_path)
        chunks = [audio[i:i + chunk_duration] for i in range(0, len(audio), chunk_duration)]
        for i, chunk in enumerate(chunks):
            chunk.export(f"{chunk_folder_path}/chunk_{i}.{audio_format}", format=audio_format)
    else:
        chunks = os.listdir(chunk_folder_path)

    result = []
    for i in range(len(chunks)):
        params = {
            "diarize": diarize,
            "call_id": request_id,
            "language": language,
            "sequence_id": i,
            "max_num_speakers": max_num_speakers,
            "min_num_speakers": min_num_speakers
        }
        t_url = f"{base_url}/transcribe?{'&'.join(f'{key}={value}' for key, value in params.items())}"
        files=[
            ('audio_file',('file',open(f"{chunk_folder_path}/chunk_{i}.{audio_format}",'rb'),'application/octet-stream'))
        ]
        response = requests.request("POST", t_url, headers={}, data={}, files=files)
        res = response.json()
        result.append(res)
        print(res)
    return result


model_size = "medium.en"
temperature = 0
beam_size = 2
nst = 0.6
try:
    rows = []
    chunk_duration = 5 * 1000
    
    config_key = f"{model_size}_b{beam_size}_t{str(temperature).replace('.', '')}_nst{str(nst).replace('.', '')}"
    for idx, audio in enumerate(audios):
        try:
            result = {
                "model": model_size,
                "temperature": temperature,
                "beam_size": beam_size,
                "no_speech_threshold": nst
            }
            audio_file_key = audio.split("/")[-1].split(".")[0]
            print("audio file key", audio_file_key)
            result["audio_id"] = audio_file_key
            request_id = generate_random_string() # any random string
            chunk_transcript = get_chunk_transcription(request_id=request_id, audio_path=audio, chunk_duration=chunk_duration, audio_file_key=audio_file_key)
            write_chunks_to_readable_file(chunk_transcript, f"{permformance_output_base_path}/{audio_file_key}.txt")
            original_transcript_text = get_original_transcript_txt(file_path=f"{original_transcript_base_path}/{audio_file_key}.txt")
            model_transcript_text = ""
            for chunk in chunk_transcript:
                for segment in chunk.get("segments"):
                    model_transcript_text += " " + segment.get("text")
            fuzzy_unmatched_score = []
            matched_words = []
            mismatched_words = []
            punctuation_errors = []
            model_transcript_text = model_transcript_text.lower().replace("okay", "ok").replace("ok", "okay")
            original_transcript_text = original_transcript_text.lower().replace("okay", "ok").replace("ok", "okay")
            original_segments = original_transcript_text.strip().split()
            transcript_segments = model_transcript_text.strip().split()
            window_size = abs(
                len(original_segments) - len(transcript_segments)
            )
            for i in range(
                min(len(original_segments), len(transcript_segments))
            ):
                word1 = transcript_segments[i]
                matched = False
                best_matched_word = ""
                max_similarity_score = 0
                for j in range(
                    max(i - window_size, 0),
                    min(i + window_size + 1, len(original_segments)),
                ):
                    word2 = original_segments[j]
                    if (
                        "okay" not in word1.lower()
                        and "ok" in word1.lower()
                    ):
                        word1 = word1.lower().replace("ok", "okay")
                    if (
                        "okay" not in word2.lower()
                        and "ok" in word2.lower()
                    ):
                        word2 = word2.lower().replace("ok", "okay")
                    if word1.lower() == word2.lower():
                        matched = True
                        break
                    else:
                        match_ratio = fuzz.ratio(
                            word1.lower(), word2.lower()
                        )
                        # if match_ratio >= 80:  avoid punctuation
                        #     matched = True
                        if match_ratio > max_similarity_score:
                            max_similarity_score = match_ratio
                            best_matched_word = word2

                og_text = " ".join(
                    [
                        original_segments[index]
                        for index in range(
                            max(i - window_size, 0),
                            min(
                                i + window_size + 1, len(original_segments)
                            ),
                        )
                    ]
                )
                ts_text = " ".join(
                    [
                        transcript_segments[index]
                        for index in range(
                            max(i - window_size, 0),
                            min(
                                i + window_size + 1,
                                len(transcript_segments),
                            ),
                        )
                    ]
                )

                if matched:
                    matched_words.append(word1)
                else:
                    segment = {}
                    segment["word"] = word1
                    segment["best_matched_word"] = best_matched_word
                    segment["max_similarity_score"] = max_similarity_score
                    segment["original"] = og_text
                    segment["transcript"] = ts_text
                    mismatched_words.append(segment)
                    fuzzy_unmatched_score.append(max_similarity_score / 100)
            

            matched_without_punc = []
            match_not_found = []
            print("mismatched", len(mismatched_words))

            for segment in mismatched_words:
                word = segment.get("word", "")
                best_match = segment.get("best_matched_word", "")
                if best_match and word.lower().replace("?", "").replace(
                    ",", ""
                ).replace(".", "") == best_match.lower().replace(
                    "?", ""
                ).replace(
                    ",", ""
                ).replace(
                    ".", ""
                ):
                    punctuation_errors.append(segment)
                else:
                    match_not_found.append(segment)
                    

            matched_wo_punc = matched_words + punctuation_errors

            result["total_words_ts"] = len(transcript_segments)
            result["total_words_og"] = len(original_segments)
            result["matched_words"] = len(matched_words)
            result["mismatched_words"] = len(mismatched_words)
            result["punctuation_errors"] = len(punctuation_errors)
            result["matched_words_wo_punc"] = len(matched_wo_punc)
            result["total_matched (%)"] = round(
                (len(matched_words) / len(transcript_segments)) * 100, 2
            )
            result["total_matched_wo_punc (%)"] = round(
                (len(matched_wo_punc) / len(transcript_segments)) * 100, 2
            )
            
            result["total_mismatched (%)"] = round((
                len(mismatched_words) / len(transcript_segments)
            ) * 100, 2)
            
            result["total_mismatched_wo_punc (%)"] = round((
                (len(mismatched_words) - len(punctuation_errors)) / len(original_segments)
            ) * 100, 2)

            result["total_mismatched_relative (%)"] = round((
                len(mismatched_words) / len(transcript_segments)
            ) * 100, 2)
            
            result["total_mismatched_wo_punc_relative (%)"] = round((
                (len(mismatched_words) - len(punctuation_errors)) / len(original_segments)
            ) * 100, 2)


            file_dir2 = f"{permformance_output_base_path}/chunk_performance_medium_en/{config_key}"
            if not os.path.exists(file_dir2):
                os.makedirs(file_dir2)
            
            mismatched_file_path = os.path.join(
                file_dir2,
                f"{audio_file_key}_{config_key}_mismatched_transcript.json",
            )
            # # Option 1: Using json.dump() to write to a file
            json_string_mismatched = json.dumps(
                mismatched_words, indent=2
            )  # The 'indent' parameter adds formatting for readability
            with open(mismatched_file_path, "w") as json_file:
                json_file.write(json_string_mismatched)

            matched_wo_punc_file = os.path.join(
                file_dir2,
                f"{audio_file_key}_{config_key}_punctuation_errors.json",
            )
            # # Option 1: Using json.dump() to write to a file
            json_string_matched_wo_punc = json.dumps(
                punctuation_errors, indent=2
            )  # The 'indent' parameter adds formatting for readability
            with open(matched_wo_punc_file, "w") as json_file:
                json_file.write(json_string_matched_wo_punc)

            
            match_not_found_file = os.path.join(
                file_dir2,
                f"{audio_file_key}_{config_key}_match_not_found.json",
            )
            # # Option 1: Using json.dump() to write to a file
            json_match_not_found = json.dumps(
                match_not_found, indent=2
            )  # The 'indent' parameter adds formatting for readability
            with open(match_not_found_file, "w") as json_file:
                json_file.write(json_match_not_found)

            transcript_file_path = os.path.join(
                file_dir2,
                f"{audio_file_key}_{config_key}_transcript.json",
            )

            # Option 1: Using json.dump() to write to a file
            chunks_json_string = json.dumps(
                chunk_transcript, indent=2
            )  # The 'indent' parameter adds formatting for readability
            with open(transcript_file_path, "w") as json_file:
                json_file.write(chunks_json_string)
            result["audio_file"] = audio
            result[
                "misatched_transcript_file"
            ] = mismatched_file_path
            result[
                "punctuation_errors_file"
            ] = matched_wo_punc_file
            result["match_not_found_file"] = match_not_found_file
            result["cosine-similarity_scikit_tfidf"] = get_cosine_similarity_tfidf(original_transcript_text, model_transcript_text)
            result["cosine-similarity_scikit_count"] = get_cosine_similarity_count(original_transcript_text, model_transcript_text)
            result["jaccard_similarity"] = get_jaccard_similarity(original_transcript_text, model_transcript_text)

            common_transcript_file = os.path.join(
                file_dir2,
                f"{audio_file_key}_{config_key}_common_transcript_file.txt",
            )

            with open(common_transcript_file, "w") as common_file:
                common_file.write(
                    "#####################  Original Script #########################"
                )
                common_file.write("\n\n")
                common_file.write(original_transcript_text)
                common_file.write("\n\n\n\n")
                common_file.write(
                    "##################### Transcribed Script #########################"
                )
                common_file.write("\n\n")
                common_file.write(model_transcript_text)

            result["common_transcript_file"] = common_transcript_file
            rows.append(result)
            print("*****************", result)
            delete_url = f"{base_url}/transcribe/{request_id}"
            response = requests.delete(delete_url)
            print(response.json())
        except Exception as e:
            print (e)
            continue
except Exception as e:
    import traceback

    traceback.print_exc()
    print(e)


try:
    df = pd.DataFrame(rows)
    # df["transcript"] = json.dumps(transcript)
    excel_file_name = f"{permformance_output_base_path}/chunk_{model_size}.xlsx"
    # Load the existing Excel file
    if os.path.exists(excel_file_name):

    # Create a new ExcelWriter with the existing file
        with pd.ExcelWriter(
            excel_file_name, engine="openpyxl", mode="a", if_sheet_exists='replace'
        ) as writer:
            # Write the DataFrame to a new sheet (you can specify the sheet name)
            df.to_excel(writer, sheet_name=config_key, index=False)
    else:
        df.to_excel(excel_file_name, sheet_name=config_key, index=False)

    print()
    print("#" * 100)
except Exception as e:
    print(e)