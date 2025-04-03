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
    "dragon", "dark", "steel", "fairy"
]

# Matrice des multiplicateurs d'après le tableau officiel des types Pokémon
type_chart = {
    "normal":   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0, 1, 1, 0.5, 1],
    "fire":     [1, 0.5, 0.5, 1, 2, 2, 1, 1, 1, 1, 1, 2, 0.5, 1, 0.5, 1, 2, 1],
    "water":    [1, 2, 0.5, 1, 0.5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 0.5, 1, 1, 1],
    "electric": [1, 1, 2, 0.5, 0.5, 1, 1, 1, 0, 2, 1, 1, 1, 1, 0.5, 1, 1, 1],
    "grass":    [1, 0.5, 2, 1, 0.5, 1, 1, 0.5, 2, 0.5, 1, 0.5, 2, 1, 0.5, 1, 0.5, 1],
    "ice":      [1, 0.5, 0.5, 1, 2, 0.5, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 0.5, 1],
    "fighting": [2, 1, 1, 1, 1, 2, 1, 0.5, 1, 0.5, 0.5, 0.5, 2, 0, 1, 2, 2, 0.5],
    "poison":   [1, 1, 1, 1, 2, 1, 1, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 1, 1, 0, 2],
    "ground":   [1, 2, 1, 2, 0.5, 1, 1, 2, 1, 0, 1, 0.5, 2, 1, 1, 1, 2, 1],
    "flying":   [1, 1, 1, 0.5, 2, 1, 2, 1, 1, 1, 1, 2, 0.5, 1, 1, 1, 0.5, 1],
    "psychic":  [1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0.5, 1, 1, 1, 1, 0, 0.5, 1],
    "bug":      [1, 0.5, 1, 1, 2, 1, 0.5, 0.5, 1, 0.5, 2, 1, 1, 0.5, 1, 2, 0.5, 0.5],
    "rock":     [1, 2, 1, 1, 1, 2, 0.5, 1, 0.5, 2, 1, 2, 1, 1, 1, 1, 0.5, 1],
    "ghost":    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 0.5, 1, 1],
    "dragon":   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0.5, 0],
    "dark":     [1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 2, 1, 1, 2, 1, 0.5, 1, 0.5],
    "steel":    [1, 0.5, 0.5, 0.5, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0.5, 2],
    "fairy":    [1, 0.5, 1, 1, 1, 1, 2, 0.5, 1, 1, 1, 1, 1, 1, 2, 2, 0.5, 1],
}

df_type_chart = pd.DataFrame(type_chart, index=type_list).T


type_list = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
    "dragon", "dark", "steel", "fairy"
]

type_to_idx = {t: i for i, t in enumerate(type_list)}

# Convertir le DataFrame précédent en matrice NumPy
type_matrix = df_type_chart.to_numpy()

#Fonctions nécessaires
def obtain_one_hot_vector_type(p) :
    """renvoie le type du pokemon au format one-hot encoding
    entrée : poke_env.environment.pokemon.Pokemon
    sortie : np.array, de longueur fixée 18"""
    vec = np.zeros(18)

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


class embedding_Player(Gen8EnvSinglePlayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = Discrete(9)  # ✅ attribut classique
    
    #Toujours mêmes valeurs de reward
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle )  :
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        moves_real_power = -np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=type_chart
                )
                moves_real_power[i] = moves_dmg_multiplier[i]*moves_base_power[i]


        #Pokemon types  
        pokemon_types = obtain_pokemon_types(battle)

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_real_power,
                pokemon_types,
            ]
        )

        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = (
            [-1] * 4 +          # real power
            [0] * 108 +         # my team types
            [0] * 108           # opponent team types
        )
        high = (
            [3] * 4 +           # real power
            [1] * 108 +         # my team types
            [1] * 108           # opponent team types
        )

        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32
        )
    
    def action_to_move(self, action: int, battle: AbstractBattle):
        order = super().action_to_move(action, battle)
        order.dynamax = False  # 🔥 désactive Dynamax pour toutes les actions
        return order
    


class NoDynamaxRandomPlayer(RandomPlayer):
    def choose_move(self, battle):
        choice = super().choose_move(battle)
        if choice.dynamax:
            choice.dynamax = False  # désactive l'option
        return choice
    

# instanciation du player
train_env_raw = embedding_Player(
    battle_format="gen8randombattle",
    opponent=RandomPlayer(battle_format="gen8randombattle"),
    start_challenging=True
)

# wrap dans DummyVecEnv (SB3 attend un vecteur d’envs, même pour un seul)
train_env = DummyVecEnv([lambda: train_env_raw])

env = train_env


model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=2.5e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.5,
    train_freq=1,
    target_update_interval=1,
    exploration_fraction=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[128, 64])
)



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