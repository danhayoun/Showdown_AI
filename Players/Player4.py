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
from Utils.embedding.type.type_embedding import *
from Utils.get.stats.get_stats import *
from stable_baselines3.common.callbacks import BaseCallback
#On va créer les fonctions pour construire les différentes composantes du state : my_team, opponent_team, etc...
MAX_HP = 714 #Blissey max


def init_my_team(battle) :
    my_team = np.zeros((72),dtype=np.float32)
    my_team_index = {}  # mapping nom court -> index (0 à 5)
    for i ,(_,p) in enumerate(battle.team.items()) :
        start = i * 12
        end = start + 12 #Pour le slicing
        alive = np.array([1], dtype=np.float32)
        type = np.array(get_type_embedding_from_pokemon(p),dtype=np.float32) #vecteur de type
        stats = get_stats(p) #vecteur de stats
        v = np.concatenate((alive,type,stats))
        my_team[start:end] = v

        my_team_index[p.species.lower()] = i  # ex: "Absol" ➜ 0

    return my_team, my_team_index




def update_my_team(battle, my_team, my_team_index):
    for _, p in battle.team.items():
        i = my_team_index[p.species.lower()]
        start = i * 12
        my_team[start] = 0.0 if p.fainted else 1.0
        my_team[start + 1] = p.current_hp/MAX_HP  # HP brut
    return my_team



def init_opponent_team(battle):
    opponent_team = np.zeros((72), dtype=np.float32)
    opponent_team_index = {}

    for i, (_, p) in enumerate(battle.opponent_team.items()):
        start = i * 12
        end = start + 12

        alive = np.array([1], dtype=np.float32)
        type_vec = np.array(get_type_embedding_from_pokemon(p), dtype=np.float32)
        stats = get_stats(p)  # hypothétique : basé sur base stats + lvl 100

        v = np.concatenate((alive, type_vec, stats))
        opponent_team[start:end] = v

        opponent_team_index[p.species.lower()] = i

    return opponent_team, opponent_team_index


def update_opponent_team(battle, opponent_team, opponent_team_index):


    for _, p in battle.opponent_team.items():
        if p.species.lower() not in opponent_team_index:
            # Nouveau Pokémon révélé
            i = len(opponent_team_index)
            if i >= 6:
                continue  # on ne peut pas avoir plus de 6 Pokémon

            start = i * 12
            end = start + 12

            alive = np.array([1], dtype=np.float32)
            type_vec = np.array(get_type_embedding_from_pokemon(p), dtype=np.float32)
            stats = get_stats(p)  # base stats + hypothèses

            v = np.concatenate((alive, type_vec, stats))
            opponent_team[start:end] = v
            opponent_team_index[p.species.lower()] = i

        # Mise à jour PV & KO
        i = opponent_team_index[p.species.lower()]
        start = i * 12
        opponent_team[start] = 0.0 if p.fainted else 1.0
        opponent_team[start + 1] = p.current_hp/MAX_HP  # ou p.current_hp_fraction * hp_max si tu l'as

    p = battle.opponent_active_pokemon
    if p.species.lower() not in opponent_team_index:
        i = len(opponent_team_index)
        start = i * 12
        end = start + 12

        alive = np.array([1], dtype=np.float32)
        type_vector = np.array(get_type_embedding_from_pokemon(p), dtype=np.float32)
        stats = get_stats(p)

        v = np.concatenate((alive, type_vector, stats))
        opponent_team[start:end] = v
        opponent_team_index[p.species.lower()] = i

    return opponent_team, opponent_team_index



class CustomTQDMCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq=500, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(total=self.total_timesteps, desc="📊 Training progress")
    
    def _on_step(self) -> bool:
        # Avance la barre
        if self.progress_bar:
            self.progress_bar.update(1)

        # Affichage toutes les X étapes
        if self.n_calls % self.check_freq == 0 and self.verbose:
            self.progress_bar.set_postfix_str(f"Timesteps: {self.num_timesteps}")
        
        return True

    def _on_training_end(self) -> None:
        if self.progress_bar:
            self.progress_bar.close()