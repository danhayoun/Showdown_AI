import os
import json
import poke_env
from poke_env.environment.pokemon import Pokemon
import torch
import poke_env
import pandas as pd
import numpy as np
from gymnasium.spaces import Space, Box, Discrete
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.data import GenData
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.random_player import RandomPlayer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from tqdm.notebook import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from IPython.display import display
import ipywidgets as widgets
import torch.nn as nn

#On charge le dictionnaire {type : embedding}
with open("json/dict_type_embeddings.json", "r", encoding="utf-8") as f:
    dict_type_embeddings = json.load(f)
###

#On charge type_idx : 
type_idx = {
    'normal': 0,
    'fire': 1,
    'water': 2,
    'electric': 3,
    'grass': 4,
    'ice': 5,
    'fighting': 6,
    'poison': 7,
    'ground': 8,
    'flying': 9,
    'psychic': 10,
    'bug': 11,
    'rock': 12,
    'ghost': 13,
    'dragon': 14,
    'dark': 15,
    'steel': 16
}
    

#Maintenant on va créer la fonction qui permet de passer de pokemon.type1 et pokemon.type2 au type au format string, puis à l'embedding, et qui renvoie l'embedding
#En bref : passer du pokemon à son type embedding

def get_type_embedding_from_pokemon(p : Pokemon) -> torch.Tensor : #Validé en test ✅✅
    """reçoit un pokemon, renvoie l'embedding de son type"""
    if p.type_2 : #Si 2 types
        if type_idx[p.type_2.name.lower()] > type_idx[p.type_1.name.lower()] :
            key_type = p.type_1.name.lower() + "_" + p.type_2.name.lower()

        else : 
            key_type = p.type_2.name.lower() + "_" + p.type_1.name.lower()

    else : #Sinon
        key_type = p.type_1.name.lower()

    return dict_type_embeddings[key_type]



