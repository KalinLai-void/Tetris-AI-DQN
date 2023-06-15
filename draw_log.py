import matplotlib.pyplot as plt
import pandas as pd
import os

dirs = [name for name in os.listdir("logs") if os.path.isdir(os.path.join("logs", name))]
dirs.sort(reverse=True)
dirs = [dirs[0]]  # take the most recent model
# dirs = [
#     'tetris-20190802-221032-ms25000-e1-ese2000-d0.99',
#     'tetris-20190802-033219-ms20000-e1-ese2000-d0.95',
# ]
#dirs = ["tetris-20230614-150528-ms25000-e3-ese2000-d0.99"]
max_scores = []
for d in dirs:
    print(f"Drawing dir '{d}'")
    log_dir = "logs/" + d
    df = pd.read_csv(f"{log_dir}/avg_scores.csv")
    steps, score = df["Step"], df["Value"]
    
    plt.figure()
    plt.plot(steps, score)
    plt.xlabel("Episodes")
    plt.ylabel("Avg. Score")
    plt.title("Training Avg. Score per 10 Episodes")
    plt.show()

    print(df)