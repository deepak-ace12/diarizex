import pandas as pd
import json
from fuzzywuzzy import fuzz
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


def get_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()

    # Compute TF-IDF vectors for the two texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute cosine similarity between the two TF-IDF vectors
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return round(cosine_sim[0][0] * 100, 2)


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
model_sizes = ["large-v2", "large-v1", "medium"]
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

                        txt = ""
                        try:
                            with open(
                                f"/Users/I1597/Downloads/Data/original_transcription/{unique_key}.txt",
                                "r",
                            ) as file:
                                file_contents = file.read()
                                for line in file_contents.splitlines():
                                    if line:
                                        txt += line.replace("P: ", " ").replace(
                                            "D: ", " "
                                        )
                        except Exception as ex:
                            print("Read Error", ex)
                            continue
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
                                ]
                            )

                        transcription_confidence_score = []
                        transcript_segments = transcript["word_segments"]
                        for word in transcript_segments:
                            transcription_confidence_score.append(word.get("score", 0))
                        result.update(
                            get_score_percentage(
                                transcription_confidence_score, "confidence"
                            )
                        )
                        result["scikit-similarity"] = get_cosine_similarity(txt, txt2)
                        result["jaccard_similarity"] = get_jaccard_similarity(txt, txt2)

                        fuzzy_unmatched_score = []
                        matched_words = []
                        mismatched_words = []
                        matched_wo_punc = []
                        original_segments = (
                            txt.replace("uh", "")
                            .replace("Uh", "")
                            .replace("uh,", "")
                            .replace("Uh,", "")
                            .replace("um", "")
                            .replace("Um", "")
                            .replace("um,", "")
                            .replace("Um,", "")
                            .strip()
                            .split()
                        )

                        for i in range(
                            min(len(original_segments), len(transcript_segments))
                        ):
                            segment = transcript_segments[i]
                            word1 = segment.get("word")
                            matched = False
                            best_matched_word = ""
                            max_similarity_score = 0
                            if word1 in [
                                "um",
                                "Um",
                                "um,",
                                "Um,",
                                "Uh",
                                "uh",
                                "Uh,",
                                "uh,",
                            ]:
                                continue
                            for j in range(
                                max(i - 10, 0), min(i + 30 + 1, len(original_segments))
                            ):
                                word2 = original_segments[j]
                                if word1.lower() == "ok":
                                    word1 = "okay"
                                if word2.lower() == "ok":
                                    word2 = "okay"
                                if word1.lower().replace("?", "").replace(
                                    ",", ""
                                ).replace(".", "") == word2.lower().replace(
                                    "?", ""
                                ).replace(
                                    ",", ""
                                ).replace(
                                    ".", ""
                                ):
                                    segment["compared_word"] = word2
                                    matched_wo_punc.append(segment)
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
                            # i = 0
                            # while i < len(transcript_segments):
                            # # for i, segment in enumerate(transcript_segments):
                            #     segment = transcript_segments[i]
                            #     word1 = segment.get("word")
                            #     if word1 in ["um", "Um", "um,", "Um,", "Uh", "uh", "Uh,", "uh,"]:
                            #         i += 1
                            #         continue
                            #     matched = False
                            #     best_matched_word = ""
                            #     max_similarity_score = 0
                            #     for k in range(max(i - 15, 0), min(i + 30 + 1, len(original_segments))):
                            #         word2 = original_segments[k]
                            #         if word1.lower() == "ok":
                            #             word1 = "okay"
                            #         if word2.lower() == "ok":
                            #             word2 = "okay"

                            #         if word1.lower().replace("?", "").replace(",", "").replace(".", "") == word2.lower().replace("?", "").replace(",", "").replace(".", ""):
                            #             segment["compared_word"] = word2
                            #             matched_wo_punc.append(segment)
                            #         if word1.lower() == word2.lower():
                            #             matched = True
                            #             break
                            #         else:
                            #             match_ratio = fuzz.ratio(word1.lower(), word2.lower())
                            #             # if match_ratio >= 80:  avoid punctuation
                            #             #     matched = True
                            #             if match_ratio > max_similarity_score:
                            #                 max_similarity_score = match_ratio
                            #                 best_matched_word = word2

                            # import ipdb; ipdb.set_trace()
                            og_text = " ".join(
                                [
                                    original_segments[index]
                                    for index in range(
                                        max(i - 10, 0),
                                        min(i + 30 + 1, len(original_segments)),
                                    )
                                ]
                            )
                            ts_text = " ".join(
                                [
                                    transcript_segments[index].get("word")
                                    for index in range(
                                        max(i - 10, 0),
                                        min(i + 30 + 1, len(transcript_segments)),
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
                        print("mismatched", len(mismatched_words))
                        for word in matched_words:
                            matched_confidence_score.append(word.get("score", 0))
                        for word in mismatched_words:
                            mismatched_confidence_score.append(word.get("score", 0))
                        for word in matched_wo_punc:
                            matched_without_punc.append(word.get("score", 0))

                        result["total_words"] = len(transcript_segments)
                        result["matched_words"] = len(matched_words)
                        result["mismatched_words"] = len(mismatched_words)
                        result.update(
                            get_score_percentage(matched_confidence_score, "matched")
                        )
                        result["matched_words_wo_punc"] = len(matched_wo_punc)
                        result.update(
                            get_score_percentage(
                                matched_without_punc, "matched_wo_punc"
                            )
                        )
                        result["total_matched (%)"] = round(
                            (len(matched_words) / len(transcript_segments)) * 100, 2
                        )
                        result.update(
                            get_score_percentage(
                                mismatched_confidence_score, "mismatched"
                            )
                        )
                        result["total_mismatched (%)"] = (
                            len(mismatched_words) / len(transcript_segments)
                        ) * 100

                        result.update(
                            get_score_percentage(
                                fuzzy_unmatched_score, "fuzzy_unmatched"
                            )
                        )

                        file_dir2 = f"/Users/I1597/Downloads/new_performance_{mapping.get(model_size)}/{config_key}"
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
                            file_dir2, f"{unique_key}_{config_key}_matched_wo_punc.json"
                        )
                        # # Option 1: Using json.dump() to write to a file
                        json_string_matched_wo_punc = json.dumps(
                            matched_wo_punc, indent=2
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
                        result["matched_wo_punc_file"] = matched_wo_punc_file.replace(
                            "/Users/I1597/Downloads/", "/"
                        )
                        rows.append(result)
                    except Exception as ex:
                        import traceback

                        traceback.print_exc()
                        print("error", unique_key, str(ex))

                df = pd.DataFrame(rows)
                # df["transcript"] = json.dumps(transcript)
                excel_file_name = f"/Users/I1597/Documents/output_{model_size}.xlsx"
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
