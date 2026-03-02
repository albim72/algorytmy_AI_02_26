import pandas as pd
import numpy as np

checkpoints = [
    (101,"Start",0),
    (102,"Start",0),
    (103,"Start",0),
    (104,"Start",0),
    (101,"Kościeliska",56),
    (102,"Kościeliska",67),
    (103,"Kościeliska",69),
    (104,"Kościeliska",23),
    (101,"Ornak",230),
    (102,"Ornak",267),
    (103,"Ornak",366),
    (104,"Ornak",88),
    (101,"Meta",428),
    (102,"Meta",546),
    (104,"Meta",145),
]

df = pd.DataFrame(checkpoints,columns=["bib","point","minute"])
df = df.sort_values(["bib","minute"])

print(df)

#liczymy czas całkowity biegu

total_times = (
    df[df["point"]=="Meta"].set_index("bib")["minute"]
)
print(total_times)
