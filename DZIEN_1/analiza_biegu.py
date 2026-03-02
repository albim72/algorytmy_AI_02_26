import pandas as pd
import numpy as np

checkpoints = [
    (101,"Start",0),
    (102,"Start",0),
    (103,"Start",0),
    (104,"Start",0),
    (105,"Start",0),
    (101,"Kościeliska",56),
    (102,"Kościeliska",67),
    (103,"Kościeliska",69),
    (104,"Kościeliska",23),
    (105,"Kościeliska",45),
    (101,"Ornak",230),
    (102,"Ornak",267),
    (103,"Ornak",366),
    (104,"Ornak",88),
    (105,"Ornak",80),
    (101,"Meta",428),
    (102,"Meta",546),
    (104,"Meta",145),
    (105,"Meta",130),
]

df = pd.DataFrame(checkpoints,columns=["bib","point","minute"])
df = df.sort_values(["bib","minute"])

print(df)

#liczymy czas całkowity biegu

total_times = (
    df[df["point"]=="Meta"].set_index("bib")["minute"]
)
print(total_times)

#średnia prędkosc
DIST_KM = 45
avg_peed = DIST_KM / (total_times/60)
print(avg_peed)

#odchylenia std czasów

mean_time = np.mean(total_times)
std_time = np.std(total_times)
print(f"średnie odchylenie: {std_time:.2f}, średni czas: {mean_time:.2f}")

#wykrywanie anomalii
z_scores = (total_times - mean_time) / std_time
print(z_scores)

#jeśli ktos ma |z| > 2 - statystycznie podejrzany
outliers = total_times[np.abs(z_scores)>2]
print(outliers)

#analiza tempa między checkpointami
df["delta"] = df.groupby("bib")["minute"].diff()
print(df)

#średnia długość segmentu
segments_stats = df.groupby("point")["delta"].mean()
print(segments_stats)

#symulacja zmęczenia
df["fatigue_index"] = np.log1p(df["minute"])
print("_"*70)
print(df)
