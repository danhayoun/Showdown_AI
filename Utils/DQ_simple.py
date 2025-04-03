from poke_env.player import Player
import torch
from stable_baselines3 import DQN
from poke_env.data import GenData
import numpy as np

class DQ_simple(Player):
    def __init__(self, model_path = "dqn_pokemon_showdown", battle_format="gen8randombattle"):
        super().__init__(battle_format=battle_format)

        # Charger le modèle DQN
        self.model = DQN.load(model_path, device="mps" if torch.backends.mps.is_available() else "cpu")
        print("📥 Modèle DQN chargé :", self.model)

    def choose_move(self, battle):
        print("👉 choose_move appelée !")
        # 🔍 Debug : Voir les moves disponibles
        print(f"🔍 Moves disponibles : {[move.id for move in battle.available_moves]}")

        # Obtenir l'observation de l'état du combat
        obs = self.embed_battle(battle)
        print("📊 Observation de l'état :", obs)

        # Transformer en format PyTorch
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        print("📊 Tensor pour le modèle :", obs_tensor)

        # Prédire l'action avec le modèle DQN
        action = int(self.model.predict(obs_tensor, deterministic=True)[0])
        print("🎯 Action choisie par DQN :", action)
        if 0 <= action < len(battle.available_moves):
            move = battle.available_moves[action]
            print(f"✅ Move choisi : {move.id}")

            order = self.create_order(move)
            print(f"📤 Ordre créé : {order}")  # Debug : voir l'ordre exact généré

            return order
        else:
            print(f"❌ Action {action} invalide, on joue un move aléatoire !")
            return self.choose_random_move(battle)
        
    def embed_battle(self, battle):
        #To define
        return 0 