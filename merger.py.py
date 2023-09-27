# add nst

import pandas as pd

excel_file_path = "/Users/I1597/Downloads/output_med_copy.xlsx"
new_column_name = "no_sound_threshold"
xls = pd.ExcelFile(excel_file_path)
sheet_names = xls.sheet_names
dfs = {sheet_name: xls.parse(sheet_name) for sheet_name in sheet_names}
for sheet_name, df in dfs.items():
    try:
        constant_value = int(sheet_name[-2:]) / 10
        df.insert(2, new_column_name, constant_value)
    except Exception as ex:
        print(sheet_name)
try:
    with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
except Exception as ex:
    print(ex)


# Merge based on file name

import pandas as pd

files = {
    "large-v1": "/Users/I1597/Downloads/new_performance_metrics/output_large-v1.xlsx",
    "medium": "/Users/I1597/Downloads/new_performance_metrics/output_medium.xlsx",
    "large-v2": "/Users/I1597/Downloads/new_performance_metrics/output_large-v2.xlsx",
}

for key, original_excel_file in files.items():
    print(original_excel_file)
    xls = pd.ExcelFile(original_excel_file)
    sheets = [
        "",
        "CAR0001",
        "CAR0002",
        "CAR0003",
        "GAS0003",
        "GAS0004",
        "GAS0007",
        "MSK0001",
        "MSK0003",
        "MSK0004",
        "RES0001",
        "RES0003",
        "RES0127",
        "RES0128",
        "MSK0042",
        "GEN0001",
    ]
    for n in range(1, len(sheets)):
        rows_list = []
        for sheet_name in xls.sheet_names:
            print(n, sheet_name)
            if sheet_name != "SheetName":
                df = pd.read_excel(original_excel_file, sheet_name=sheet_name)
                rows_list.append(df.iloc[n - 1])
        result_df = pd.DataFrame(rows_list)
        if n == 1:
            with pd.ExcelWriter(
                f"/Users/I1597/Downloads/new_performance_metrics/output_{key}_combined.xlsx",
                engine="openpyxl",
            ) as writer:
                result_df.to_excel(writer, sheet_name=sheets[n], index=False)
        else:
            with pd.ExcelWriter(
                f"/Users/I1597/Downloads/new_performance_metrics/output_{key}_combined.xlsx",
                engine="openpyxl",
                mode="a",
            ) as writer:
                result_df.to_excel(writer, sheet_name=sheets[n], index=False)
        print()
        print("#" * 100)


# merge based on models
import pandas as pd

med = "/Users/I1597/Downloads/new_performance_metrics/output_medium_combined.xlsx"
v1 = "/Users/I1597/Downloads/new_performance_metrics/output_large-v1_combined.xlsx"
v2 = "/Users/I1597/Downloads/new_performance_metrics/output_large-v2_combined.xlsx"

xls_med = pd.ExcelFile(med)
xls_v1 = pd.ExcelFile(v1)
xls_v2 = pd.ExcelFile(v2)


sheets = [
    "CAR0001",
    "CAR0002",
    "CAR0003",
    "GAS0003",
    "GAS0004",
    "GAS0007",
    "MSK0001",
    "MSK0003",
    "MSK0004",
    "RES0001",
    "RES0003",
    "RES0127",
    "RES0128",
    "MSK0042",
    "GEN0001",
]
for sheet_name in sheets:
    rows_list = []
    for n in range(1, 49):
        print(n)
        df_med = pd.read_excel(med, sheet_name=sheet_name)
        rows_list.append(df_med.iloc[n - 1])

        df_v1 = pd.read_excel(v1, sheet_name=sheet_name)
        rows_list.append(df_v1.iloc[n - 1])

        df_v2 = pd.read_excel(v2, sheet_name=sheet_name)
        rows_list.append(df_v2.iloc[n - 1])

    result_df = pd.DataFrame(rows_list)
    if sheet_name == "CAR0001":
        with pd.ExcelWriter(
            "/Users/I1597/Downloads/new_performance_metrics/output_med_v1_v2_combined.xlsx",
            engine="openpyxl",
        ) as writer:
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(
            "/Users/I1597/Downloads/new_performance_metrics/output_med_v1_v2_combined.xlsx",
            engine="openpyxl",
            mode="a",
        ) as writer:
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print()
        print("#" * 100)


from pydub import AudioSegment

audio = AudioSegment.from_file(
    "/Users/I1597/Downloads/Data/Audio Recordings/GAS0007.mp3"
)
segment_length = 100  # in milliseconds
segments = []
for start in range(0, len(audio), segment_length):
    end = start + segment_length
    segment = audio[start:end]
    text = f"{start/1000} - {end/1000}: rms: {segment.rms}, dBFS: {round(segment.dBFS, 2)}, max_dBFS: {round(segment.max_dBFS, 2)}, channels: {segment.channels}"
    segments.append(text)
