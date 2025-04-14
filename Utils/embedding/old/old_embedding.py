import poke_env
import pandas as pd
import numpy as np
from gymnasium.spaces import Space, Box
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.data import GenData
import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
import torch
from gymnasium.spaces import Discrete, Box
from poke_env.player.random_player import RandomPlayer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from tqdm.notebook import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from IPython.display import display
import ipywidgets as widgets
import torch.nn as nn

#Table des types, forme matricielle
type_list = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
    "dragon", "dark", "steel", 
]

# Matrice des multiplicateurs d'après le tableau officiel des types Pokémon
type_chart = {
    "normal":   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0, 1, 1, 0.5],
    "fire":     [1, 0.5, 0.5, 1, 2, 2, 1, 1, 1, 1, 1, 2, 0.5, 1, 0.5, 1, 2],
    "water":    [1, 2, 0.5, 1, 0.5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 0.5, 1, 1],
    "electric": [1, 1, 2, 0.5, 0.5, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0.5, 1, 1],
    "grass":    [1, 0.5, 2, 1, 0.5, 1, 1, 0.5, 2, 0.5, 1, 0.5, 2, 1, 0.5, 1, 0.5],
    "ice":      [1, 0.5, 0.5, 1, 2, 0.5, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 0.5],
    "fighting": [2, 1, 1, 1, 1, 2, 1, 0.5, 1, 0.5, 0.5, 0.5, 2, 0, 1, 2, 2],
    "poison":   [1, 1, 1, 1, 2, 1, 1, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 1, 1, 0],
    "ground":   [1, 2, 1, 2, 0.5, 1, 1, 2, 1, 0, 1, 0.5, 2, 1, 1, 1, 2],
    "flying":   [1, 1, 1, 0.5, 2, 1, 2, 1, 1, 1, 1, 2, 0.5, 1, 1, 1, 0.5],
    "psychic":  [1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0.5, 1, 1, 1, 1, 0, 0.5],
    "bug":      [1, 0.5, 1, 1, 2, 1, 0.5, 0.5, 1, 0.5, 2, 1, 1, 0.5, 1, 2, 0.5],
    "rock":     [1, 2, 1, 1, 1, 2, 0.5, 1, 0.5, 2, 1, 2, 1, 1, 1, 1, 0.5],
    "ghost":    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 0.5, 1],
    "dragon":   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0.5],
    "dark":     [1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 2, 1, 1, 2, 1, 0.5, 1],
    "steel":    [1, 0.5, 0.5, 0.5, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0.5],
}

df_type_chart = pd.DataFrame(type_chart, index=type_list).T


type_list = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
    "dragon", "dark", "steel"
]

type_to_idx = {t: i for i, t in enumerate(type_list)}

# Convertir le DataFrame précédent en matrice NumPy
type_matrix = df_type_chart.to_numpy()

#Fonctions nécessaires
def obtain_one_hot_vector_type(p) :
    """renvoie le type du pokemon au format one-hot encoding
    entrée : poke_env.environment.pokemon.Pokemon
    sortie : np.array, de longueur fixée 17"""
    vec = np.zeros(17)

    if p.type_1 : 
        vec[type_to_idx[p.type_1.name.lower()]] = 1.0
    if p.type_2 :
        vec[type_to_idx[p.type_2.name.lower()]] = 1.0
    return vec

def obtain_pokemon_types(battle):

    # Encode ta team
    vectors_my_team = [
        obtain_one_hot_vector_type(p) if not p.fainted else np.zeros(len(type_to_idx), dtype=np.float32)
        for _, p in battle.team.items()
    ]
    while len(vectors_my_team) < 6:
        vectors_my_team.append(np.zeros(len(type_to_idx), dtype=np.float32))
    my_team_type = np.concatenate(vectors_my_team)

    # Encode la team adverse
    vectors_opponent_team = [
        obtain_one_hot_vector_type(p) if not p.fainted else np.zeros(len(type_to_idx), dtype=np.float32)
        for _, p in battle.opponent_team.items()
    ]
    while len(vectors_opponent_team) < 6:
        vectors_opponent_team.append(np.zeros(len(type_to_idx), dtype=np.float32))
    opponent_team_type = np.concatenate(vectors_opponent_team)

    # Concatène les deux parties
    pokemon_types = np.concatenate([my_team_type, opponent_team_type])
    
    return pokemon_types


# Initialiser GenData pour la génération souhaitée (par exemple, génération 8)
gen_data = GenData.from_gen(8)

# Accéder au tableau des types
type_chart = gen_data.type_chart





