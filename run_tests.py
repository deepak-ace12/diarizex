import pandas as pd
import requests
import json
from fuzzywuzzy import fuzz
import numpy as np
import os

rows = []

audios = [
    "/Users/I1597/Downloads/Data/Audio Recordings/CAR0001.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/CAR0002.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/CAR0003.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/CAR0004.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/CAR0005.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/DER0001.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GAS0001.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GAS0002.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GAS0003.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GAS0004.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GAS0005.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GAS0007.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/GEN0001.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0001.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0003.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0004.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0005.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0006.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0007.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0008.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0009.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0010.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0011.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0012.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0013.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0014.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0015.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0016.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0017.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0018.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0019.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0020.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0021.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0022.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0023.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0024.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0025.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0001.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0002.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0003.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0004.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0005.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0006.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0007.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0008.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0009.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0010.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0026.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0027.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0028.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0029.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0031.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0032.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0033.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0034.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0035.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0036.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0037.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0038.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0039.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0040.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0041.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0042.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0043.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0044.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0045.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0046.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0049.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/MSK0050.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0001.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0002.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0003.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0004.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0005.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0006.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0007.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0008.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0009.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0010.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0011.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0012.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0013.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0014.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0015.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0016.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0017.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0018.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0019.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0020.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0021.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0022.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0023.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0024.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0025.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0026.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0027.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0028.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0029.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0030.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0031.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0032.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0033.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0034.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0035.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0036.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0037.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0038.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0039.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0042.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0043.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0044.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0045.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0046.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0047.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0048.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0049.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0050.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0051.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0052.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0053.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0054.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0055.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0056.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0057.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0058.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0059.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0060.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0061.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0062.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0063.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0064.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0065.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0066.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0067.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0068.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0069.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0070.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0071.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0072.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0073.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0074.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0075.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0076.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0077.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0078.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0079.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0080.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0081.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0082.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0083.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0084.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0085.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0086.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0087.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0088.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0089.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0090.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0091.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0092.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0093.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0094.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0095.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0096.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0097.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0098.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0099.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0100.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0101.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0102.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0103.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0104.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0105.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0106.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0107.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0108.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0109.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0110.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0111.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0112.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0113.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0114.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0116.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0117.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0118.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0119.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0120.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0121.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0122.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0123.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0124.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0125.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0126.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0127.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0128.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0129.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0130.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0131.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0132.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0133.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0134.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0135.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0136.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0137.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0138.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0139.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0140.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0141.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0142.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0143.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0144.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0145.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0146.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0147.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0148.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0149.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0150.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0151.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0152.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0153.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0154.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0155.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0156.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0158.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0159.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0160.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0161.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0162.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0163.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0164.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0165.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0166.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0167.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0168.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0169.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0170.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0171.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0172.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0173.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0174.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0175.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0176.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0177.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0178.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0179.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0180.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0181.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0182.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0183.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0184.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0185.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0186.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0187.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0188.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0189.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0190.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0191.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0192.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0193.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0194.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0195.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0196.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0197.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0198.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0199.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0200.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0201.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0202.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0203.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0204.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0205.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0206.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0207.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0208.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0209.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0210.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0211.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0212.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0213.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0214.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0215.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0216.mp3",
    "/Users/I1597/Downloads/Data/Audio Recordings/RES0217.mp3",
]

audios = ["/Users/I1597/Downloads/Data/Audio Recordings/GAS0007.mp3"]

# Example lists


def get_score_percentage(confidence_score, key):
    arr = np.array(confidence_score)
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
model_sizes = ["medium", "large-v1", "large_v2"]
nsts = [0.2, 0.4, 0.6, 0.8]


file_dir = "/"
config_key = "temp"
for idx, audio in enumerate(audios):
    print(idx)
    unique_key = audio.split("/")[-1].split(".")[0]
    url = f"http://3.84.116.137:8000/transcribe?unique_key={unique_key}"
    files = [("audio_file", ("file", open(audio, "rb"), "application/octet-stream"))]
    response = requests.request("POST", url, headers={}, data={}, files=files)
    result = response.json()
    transcript = result.pop("transcript")
    config_key = f"{result.get('model')}_b{result.get('beam_size')}_sr{result.get('sample_rate')}_t1_nst0"
    file_dir = f"/Users/I1597/Downloads/performance/{config_key}"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_path = os.path.join(file_dir, "transcript.json")

    # Option 1: Using json.dump() to write to a file
    json_string = json.dumps(
        transcript, indent=2
    )  # The 'indent' parameter adds formatting for readability
    with open(file_path, "w") as json_file:
        json_file.write(json_string)
    result["transcript_file"] = file_path
    result["audio_file"] = audio

    with open(
        f"/Users/I1597/Downloads/Data/original_transcription/{unique_key}.txt", "r"
    ) as file:
        file_contents = file.read()

    txt = ""
    for line in file_contents.splitlines():
        if line:
            txt += line.replace("P: ", " ").replace("D: ", " ")

    fuzzy_unmatched_score = []
    matched_words = []
    mismatched_words = []
    transcript_segments = transcript.get("word_segments")
    original_segments = txt.strip().split()
    for i, segment in enumerate(transcript_segments):
        word1 = segment.get("word")
        matched = False
        best_matched_word = ""
        max_similarity_score = 0

        for j in range(max(i - 3, 0), min(i + 3 + 1, len(original_segments))):
            word2 = original_segments[j]
            if word1.lower() == word2.lower():
                matched = True
                i = j
                break
            else:
                og_text = " ".join(
                    [
                        original_segments[index]
                        for index in range(
                            max(i - 2, 0), min(i + 2 + 1, len(original_segments))
                        )
                    ]
                )
                ts_text = " ".join(
                    [
                        transcript_segments[index].get("word")
                        for index in range(
                            max(i - 2, 0), min(i + 2 + 1, len(transcript_segments))
                        )
                    ]
                )
                match_ratio = fuzz.ratio(word1.lower(), word2.lower())
                # if match_ratio >= 80:  avoid punctuation
                #     matched = True
                if match_ratio > max_similarity_score:
                    max_similarity_score = match_ratio
                    best_matched_word = word2

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

    for word in matched_words:
        matched_confidence_score.append(word.get("score", 0))
    for word in mismatched_words:
        mismatched_confidence_score.append(word.get("score", 0))

    result.update(get_score_percentage(matched_confidence_score, "matched"))
    result["total_matched (%)"] = (len(matched_words) / len(transcript_segments)) * 100
    result.update(get_score_percentage(mismatched_confidence_score, "mismatched"))
    result["total_mismatched (%)"] = (
        len(mismatched_words) / len(transcript_segments)
    ) * 100

    result.update(get_score_percentage(fuzzy_unmatched_score, "fuzzy_unmatched"))

    mismatched_file_path = os.path.join(file_dir, "mismatched_transcript.json")
    # Option 1: Using json.dump() to write to a file
    json_string_mismatched = json.dumps(
        mismatched_words, indent=2
    )  # The 'indent' parameter adds formatting for readability
    with open(mismatched_file_path, "w") as json_file:
        json_file.write(json_string_mismatched)

    rows.append(result)


df = pd.DataFrame(rows)
# df["transcript"] = json.dumps(transcript)
excel_file = f"{file_dir}/output.xlsx"
df.to_excel(excel_file, sheet_name=config_key, index=False)
