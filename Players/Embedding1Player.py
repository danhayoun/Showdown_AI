from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import torch
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data import GenData
from poke_env.player.battle_order import BattleOrder
import numpy as np
import torch
from gymnasium.spaces import Discrete, Box
import pandas as pd
from gymnasium.spaces import Space, Box
from poke_env.player import Player

################## Données nécessaires, table des types ########################################################################
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


##################################### Fonctions nécessaires ######################################################
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

#################################### CLASS PLAYER ########################################################################
#(bot et vs joueur)

class EmbeddingTestPlayervsBot(Gen8EnvSinglePlayer):

    def __init__(self, model_path="embedding_1", **kwargs):
        super().__init__(**kwargs)
        self.model = DQN.load(model_path, device="mps" if torch.backends.mps.is_available() else "cpu")
        print(f"📥 Modèle chargé depuis {model_path}")

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
    
    def action_to_move(self, action, battle):
        moves = battle.available_moves
        switches = battle.available_switches
        total_actions = len(moves) + len(switches)

        #print(f"🔢 DQN → action={action} | #moves={len(moves)} | #switches={len(switches)} | total={total_actions}")

        if 0 <= action < len(moves):
            return self.create_order(moves[action])
        elif len(moves) <= action < total_actions:
            return self.create_order(switches[action - len(moves)])
        else:
            #print("❌ Action hors bornes ! Fallback sur move aléatoire")
            return self.choose_random_move(battle)
        
    def predict(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_values = self.model.q_net(obs_tensor)
        print(f"📊 Q-values : {q_values.detach().numpy().flatten()}")
        action = int(torch.argmax(q_values).item())
        return action
    





class EmbeddingTestPlayervsHuman(Player):

    def __init__(self,account_configuration, model_path="embedding_1",battle_format="gen8randombattle"):
        super().__init__(account_configuration=account_configuration,battle_format=battle_format)
        self.model = DQN.load(model_path, device="mps" if torch.backends.mps.is_available() else "cpu")
        print(f"📥 Modèle chargé depuis {model_path}")

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
    
    def action_to_move(self, action, battle):
        moves = battle.available_moves
        switches = battle.available_switches
        total_actions = len(moves) + len(switches)

        #print(f"🔢 DQN → action={action} | #moves={len(moves)} | #switches={len(switches)} | total={total_actions}")

        if 0 <= action < len(moves):
            return self.create_order(moves[action])
        elif len(moves) <= action < total_actions:
            return self.create_order(switches[action - len(moves)])
        else:
            #print("❌ Action hors bornes ! Fallback sur move aléatoire")
            return self.choose_random_move(battle)
        
    def predict(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_values = self.model.q_net(obs_tensor)
        print(f"📊 Q-values : {q_values.detach().numpy().flatten()}")
        action = int(torch.argmax(q_values).item())
        return action
    
    def choose_move(self, battle):
            #print("👉 choose_move appelée !")
            # 🔍 Debug : Voir les moves disponibles
            #print(f"🔍 Moves disponibles : {[move.id for move in battle.available_moves]}")

            # Obtenir l'observation de l'état du combat
            obs = self.embed_battle(battle)
            #print("📊 Observation de l'état :", obs)

            # Transformer en format PyTorch
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            #print("📊 Tensor pour le modèle :", obs_tensor)

            # Prédire l'action avec le modèle DQN
            action = int(self.model.predict(obs_tensor, deterministic=True)[0])
            #print("🎯 Action choisie par DQN :", action)
            if 0 <= action < len(battle.available_moves):
                move = battle.available_moves[action]
                #print(f"✅ Move choisi : {move.id}")

                order = self.create_order(move)
                order.dynamax = False
                #print(f"📤 Ordre créé : {order}")  # Debug : voir l'ordre exact généré

                return order
            else:
                #print(f"❌ Action {action} invalide, on joue un move aléatoire !")
                return self.choose_random_move(battle)