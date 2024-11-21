import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

class Analyser:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.players = self.data.iloc[0, 3:].dropna().values  
        self.scores = self.data.iloc[1:, 3:].apply(pd.to_numeric, errors='coerce')  

    def player_scores(self, name):
        if name in self.players:
            index = list(self.players).index(name)
            scores = self.scores.iloc[:, index].dropna()
            return list(zip(scores.index, scores.values))  
        else:
            raise ValueError(f"'{name}' is not in the data lmoao check again")
        
    def player_score(self, name, game_index):
        if name not in self.players:
            raise ValueError(f"'{name}' is not in the data")
        if game_index < 0 or game_index >= len(self.scores):
            raise ValueError(f"index {game_index} is out of range")
        return self.scores.iloc[game_index][np.where(self.players == name)[0][0]]
        
    def game_ranking(self, game_index):
        # use player_score method to get the score of each player for the game_index
        # sort the scores in descending order
        # return the sorted scores
        if game_index < 0 or game_index >= len(self.scores):
            raise ValueError(f"index {game_index} is out of range")
        ranking = []
        for player in self.players:
            score = self.player_score(player, game_index)
            if not pd.isna(score):
                ranking.append((player, score))
        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        ranking = [x for x in ranking if x[0] not in ["Avg", "stddev"]]
        return ranking
    
    def game_result(self, game_index):
        if game_index < 0 or game_index >= len(self.scores):
            raise ValueError(f"index {game_index} is out of range !!!!!!")
        return self.scores.iloc[game_index].dropna().values
    
    
    def player_score_analysis(self, name, plot=False):
        if name not in self.players:
            raise ValueError(f"'{name}' not in the sheets")
        
        player_scores = self.player_score(name)
        
        # Plot if requested
        if plot:
            # perform linear regression for the line of best fit
            reg = LinearRegression().fit(
                np.array(range(len(player_scores))).reshape(-1, 1),
                np.array([score for _, score in player_scores]).reshape(-1, 1)
            )
            best_fit_line = reg.predict(np.array(range(len(player_scores))).reshape(-1, 1))
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(len(player_scores)),
                [score for _, score in player_scores],
                label="Scores",
                alpha=0.7,
                linestyle='-',
                marker='o',  # Add points as dots
            )
            plt.plot(
                range(len(player_scores)),
                best_fit_line.flatten(),
                label="Line of Best Fit",
                color="blue",
                linestyle="--",
            )
            plt.title(f"Score Analysis for {name}")
            plt.xlabel("Games")
            plt.ylabel("Scores")
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        return player_scores

    def player_zscore_analysis(self, name, plot=False, distribution=False):
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

        # Plot if requested
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
        
        if distribution:
            _, z_values = zip(*z_scores)
            plt.figure(figsize=(8, 6))
            plt.hist(z_values, bins=15, alpha=0.7, color="purple", edgecolor="black")
            plt.title(f"Z-Score Distribution for {name}")
            plt.xlabel("Z-Score")
            plt.ylabel("Frequency")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        
        return z_scores
