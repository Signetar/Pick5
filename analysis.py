import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Analyser:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.players = self.data.iloc[0, 3:].dropna().values  
        self.scores = self.data.iloc[1:, 3:].apply(pd.to_numeric, errors='coerce')  

    def player_score(self, name):
        if name in self.players:
            index = list(self.players).index(name)
            scores = self.scores.iloc[:, index].dropna()
            return list(zip(scores.index, scores.values))  
        else:
            raise ValueError(f"'{name}' is not in the data lmoao check again")
        
    def game_result(self, game_index):
        if game_index < 0 or game_index >= len(self.scores):
            raise ValueError(f"index {game_index} is out of range !!!!!!")
        return self.scores.iloc[game_index].dropna().values

    def player_zscore_analysis(self, name, plot=False):
        if name not in self.players:
            raise ValueError(f"'{name}' not in the sheets")
        
        player_scores = self.player_score(name)
        averages = self.player_score("Avg")
        stddevs = self.player_score("stddev")
        
        avg_dict = dict(averages)
        stddev_dict = dict(stddevs)
        
        z_scores = []
        for game_index, score in player_scores:
            if game_index in avg_dict and game_index in stddev_dict and stddev_dict[game_index] > 0:
                z = (score - avg_dict[game_index]) / stddev_dict[game_index]
                z_scores.append((game_index, z))
        
        if plot:
            games, z_values = zip(*z_scores)
            
            # Perform linear regression for the line of best fit
            games_array = np.array(games).reshape(-1, 1)
            z_values_array = np.array(z_values).reshape(-1, 1)
            reg = LinearRegression().fit(games_array, z_values_array)
            best_fit_line = reg.predict(games_array)
            
            plt.figure(figsize=(10, 6))
            plt.plot(
                games,
                z_values,
                label="Z-Scores",
                alpha=0.7,
                linestyle='-',
                marker='o',  # Add points as dots
            )
            plt.plot(
                games,
                best_fit_line.flatten(),
                label="Line of Best Fit",
                color="blue",
                linestyle="--",
            )
            plt.axhline(0, color='gray', linestyle='-', label="Mean (Z=0)")
            plt.axhline(2, color='red', linestyle='--', label="Z=+2 Threshold")
            plt.axhline(-2, color='red', linestyle='--', label="Z=-2 Threshold")
            plt.title(f"Z-Score Analysis for {name}")
            plt.xlabel("Games")
            plt.ylabel("Z-Scores")
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        return z_scores

