import pandas as pd
import json
from fuzzywuzzy import fuzz
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pydub import AudioSegment


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

# Example lists

stop_words = set(stopwords.words("english"))
nltk.download("punkt")
nltk.download("stopwords")


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


def get_score_percentage(confidence_score, key):
    arr = np.array(confidence_score)
    if len(arr) > 0:
        percentage_above_90 = np.sum(arr >= 0.9) / len(arr) * 100
        percentage_above_80 = np.sum(arr >= 0.8) / len(arr) * 100
        percentage_above_70 = np.sum(arr >= 0.7) / len(arr) * 100
        percentage_above_60 = np.sum(arr >= 0.6) / len(arr) * 100
        percentage_above_50 = np.sum(arr >= 0.5) / len(arr) * 100
        percentage_below_50 = np.sum(arr < 0.5) / len(arr) * 100
        median = np.median(arr)
        return {
            f"{key}_score_above_0.9 (%)": round(percentage_above_90, 2),
            f"{key}_score_above_0.8 (%)": round(percentage_above_80, 2),
            f"{key}_score_above_0.7 (%)": round(percentage_above_70, 2),
            f"{key}_score_above_0.6 (%)": round(percentage_above_60, 2),
            f"{key}_score_above_0.5 (%)": round(percentage_above_50, 2),
            f"{key}_score_below_0.5 (%)": round(percentage_below_50, 2),
            f"{key}_median": round(median, 2),
        }


temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
model_sizes = ["large-v1"]
mapping = {"large-v2": "v2", "large-v1": "v1", "medium": "med"}

nsts = [0.2, 0.4, 0.6, 0.8]
beam_sizes = [1, 5]
for model_size in model_sizes:
    for temperature in temperatures:
        for beam_size in beam_sizes:
            for nst in nsts:
                rows = []
                for idx, audio in enumerate(audios):
                    try:
                        print(idx)
                        audio_file = AudioSegment.from_file(audio)
                        unique_key = audio.split("/")[-1].split(".")[0]
                        result = {
                            "model": model_size,
                            "temperature": temperature,
                            "nst": nst,
                            "beam_size": beam_size,
                            "file_unique_key": unique_key,
                            "batch_size": 32,
                            "device": "cuda",
                            "frame_rate": 16000,
                            "file_format": "mp3",
                            "duration": round(audio_file.duration_seconds, 2),
                        }

                        fillers = ["um,", "Um,", "um", "Um", "Uh,", "uh,", "Uh", "uh"]

                        txt = ""
                        try:
                            with open(
                                f"/Users/I1597/Downloads/Data/original_transcription/{unique_key}.txt",
                                "r",
                            ) as file:
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
                        except Exception as ex:
                            print("Read Error", ex)
                            continue
                        txt = txt.replace("  ", " ")
                        config_key = f"{model_size}_b{beam_size}_sr16000_t{str(temperature).replace('.', '')}_nst{str(nst).replace('.', '')}"
                        file_dir = f"/Users/I1597/Downloads/performance_{mapping.get(model_size)}/{config_key}"
                        file_path = os.path.join(
                            file_dir, f"{unique_key}_{config_key}_transcript.json"
                        )
                        txt2 = ""
                        transcript = ""
                        with open(file_path, "r") as json_file:
                            transcript = json.load(json_file)
                            txt2 = " ".join(
                                [
                                    segment.get("word", "")
                                    for segment in transcript.get("word_segments", [])
                                    if segment.get("word") not in fillers
                                ]
                            )

                        transcription_confidence_score = []

                        transcript_segments = [
                            segment
                            for segment in transcript["word_segments"]
                            if segment.get("word") not in fillers
                        ]
                        for word in transcript_segments:
                            transcription_confidence_score.append(word.get("score", 0))
                        result.update(
                            get_score_percentage(
                                transcription_confidence_score, "confidence"
                            )
                        )

                        result["cosine-similarity_scikit_tfidf"] = get_cosine_similarity_tfidf(txt, txt2)
                        result["cosine-similarity_scikit_count"] = get_cosine_similarity_count(txt, txt2)
                        result["jaccard_similarity"] = get_jaccard_similarity(txt, txt2)

                        fuzzy_unmatched_score = []
                        matched_words = []
                        mismatched_words = []
                        punctuation_errors = []
                        original_segments = txt.strip().split()

                        window_size = abs(
                            len(original_segments) - len(transcript_segments)
                        )
                        for i in range(
                            min(len(original_segments), len(transcript_segments))
                        ):
                            segment = transcript_segments[i]
                            word1 = segment.get("word", "")
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
                                    transcript_segments[index].get("word")
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
                                matched_words.append(segment)
                            else:
                                segment["best_matched_word"] = best_matched_word
                                segment["max_similarity_score"] = max_similarity_score
                                segment["original"] = og_text
                                segment["transcript"] = ts_text
                                mismatched_words.append(segment)
                                fuzzy_unmatched_score.append(max_similarity_score / 100)

                        matched_confidence_score = []
                        mismatched_confidence_score = []
                        matched_without_punc = []
                        mismatched_dbfs = []
                        matched_dbfs = []
                        print("mismatched", len(mismatched_words))
                        for segment in matched_words:
                            matched_confidence_score.append(segment.get("score", 0))
                            try:
                                cut_audio = audio_file[segment.get("start", 0)*1000: segment.get("end", 0)*1000]
                                if cut_audio.rms > 0:
                                    dbfs = round(cut_audio.dBFS, 2)
                                    matched_dbfs.append(dbfs)
                            except Exception as ex:
                                print(ex)

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
                                try:
                                    cut_audio = audio_file[segment.get("start", 0)*1000: segment.get("end", 0)*1000]
                                    if cut_audio.rms > 0:
                                        dbfs = round(cut_audio.dBFS, 2)
                                        matched_dbfs.append(dbfs)
                                except Exception as ex:
                                    print(ex)
                            else:
                                try:
                                    cut_audio = audio_file[segment.get("start", 0)*1000: segment.get("end", 0)*1000]
                                    if cut_audio.rms > 0:
                                        dbfs = round(cut_audio.dBFS, 2)
                                        mismatched_dbfs.append(dbfs)
                                except Exception as ex:
                                    print(ex)
                            mismatched_confidence_score.append(segment.get("score", 0))

                        matched_wo_punc = matched_words + punctuation_errors

                        for segment in matched_wo_punc:
                            matched_without_punc.append(segment.get("score", 0))

                        result["total_words_ts"] = len(transcript_segments)
                        result["total_words_og"] = len(original_segments)
                        result["matched_words"] = len(matched_words)
                        result["mismatched_words"] = len(mismatched_words)
                        result["punctuation_errors"] = len(punctuation_errors)
                        result["matched_words_wo_punc"] = len(matched_wo_punc)
                        if mismatched_dbfs:
                            arr = np.array(mismatched_dbfs)
                            low_decibels = (np.sum(arr <= -40) / len(arr)) * 100
                            result["mismatched_words_wo_punc_dbfs_lt_40 (%)"] = round(low_decibels, 2)
                        if matched_dbfs:
                            arr = np.array(matched_dbfs)
                            low_decibels =  (np.sum(arr <= -40) / len(arr)) * 100
                            result["matched_words_wo_punc_dbfs_lt_40 (%)"] = round(low_decibels, 2)

                        result.update(
                            get_score_percentage(matched_confidence_score, "matched")
                        )
                        result.update(
                            get_score_percentage(
                                matched_without_punc, "matched_wo_punc"
                            )
                        )
                        result["total_matched (%)"] = round(
                            (len(matched_words) / len(transcript_segments)) * 100, 2
                        )
                        result["total_matched_wo_punc (%)"] = round(
                            (len(matched_wo_punc) / len(transcript_segments)) * 100, 2
                        )
                        result.update(
                            get_score_percentage(
                                mismatched_confidence_score, "mismatched"
                            )
                        )
                        result["total_mismatched (%)"] = round((
                            len(mismatched_words) / len(transcript_segments)
                        ) * 100, 2)
                        
                        result["total_mismatched_wo_punc (%)"] = round((
                            (len(mismatched_words) - len(punctuation_errors)) / len(transcript_segments)
                        ) * 100, 2)

                        result.update(
                            get_score_percentage(
                                fuzzy_unmatched_score, "fuzzy_unmatched"
                            )
                        )

                        file_dir2 = f"/Users/I1597/Downloads/new_performance_metrics/new_performance_{mapping.get(model_size)}/{config_key}"
                        if not os.path.exists(file_dir2):
                            os.makedirs(file_dir2)
                        mismatched_file_path = os.path.join(
                            file_dir2,
                            f"{unique_key}_{config_key}_mismatched_transcript.json",
                        )
                        # # Option 1: Using json.dump() to write to a file
                        json_string_mismatched = json.dumps(
                            mismatched_words, indent=2
                        )  # The 'indent' parameter adds formatting for readability
                        with open(mismatched_file_path, "w") as json_file:
                            json_file.write(json_string_mismatched)

                        matched_wo_punc_file = os.path.join(
                            file_dir2,
                            f"{unique_key}_{config_key}_punctuation_errors.json",
                        )
                        # # Option 1: Using json.dump() to write to a file
                        json_string_matched_wo_punc = json.dumps(
                            punctuation_errors, indent=2
                        )  # The 'indent' parameter adds formatting for readability
                        with open(matched_wo_punc_file, "w") as json_file:
                            json_file.write(json_string_matched_wo_punc)

                        result["transcript_file"] = file_path.replace(
                            "/Users/I1597/Downloads/", "/"
                        )
                        result["audio_file"] = audio.replace(
                            "/Users/I1597/Downloads/", "/"
                        )
                        result[
                            "misatched_transcript_file"
                        ] = mismatched_file_path.replace("/Users/I1597/Downloads/", "/")
                        result[
                            "punctuation_errors_file"
                        ] = matched_wo_punc_file.replace("/Users/I1597/Downloads/", "/")

                        common_transcript_file = os.path.join(
                            file_dir2,
                            f"{unique_key}_{config_key}_common_transcript_file.txt",
                        )

                        with open(common_transcript_file, "w") as common_file:
                            common_file.write(
                                "#####################  Original Script #########################"
                            )
                            common_file.write("\n\n")
                            common_file.write(txt)
                            common_file.write("\n\n\n\n")
                            common_file.write(
                                "##################### Transcribed Script #########################"
                            )
                            common_file.write("\n\n")
                            common_file.write(txt2)

                        result["common_transcript_file"] = common_transcript_file
                        rows.append(result)
                    except Exception as ex:
                        import traceback

                        traceback.print_exc()
                        print("error", unique_key, str(ex))

                df = pd.DataFrame(rows)
                # df["transcript"] = json.dumps(transcript)
                excel_file_name = f"/Users/I1597/Downloads/new_performance_metrics/output_{model_size}.xlsx"
                # Load the existing Excel file
                if os.path.exists(excel_file_name):

                    # Create a new ExcelWriter with the existing file
                    with pd.ExcelWriter(
                        excel_file_name, engine="openpyxl", mode="a"
                    ) as writer:
                        # Write the DataFrame to a new sheet (you can specify the sheet name)
                        df.to_excel(writer, sheet_name=config_key, index=False)
                else:
                    df.to_excel(excel_file_name, sheet_name=config_key, index=False)

                print()
                print("#" * 100)
