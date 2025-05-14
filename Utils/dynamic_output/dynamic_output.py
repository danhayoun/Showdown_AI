
import poke_env
import numpy as np
from stable_baselines3.dqn.policies import DQNPolicy



def get_valid_actions(battle) -> tuple[list[int], np.ndarray]:
    """
    Retourne la liste des IDs d’actions valides (0-8)
    et un masque binaire de longueur 9.
    """
    valid_action_ids = []
    action_ids_ref_dict = {}
    # Récupérer le Pokémon actif
    active = battle.active_pokemon
    if not active:
        for i, p in enumerate(battle.available_switches):
            if not p.fainted:
                if 0 <= i <= 4:  # Banc de 2 à 6
                    action_id = 4 + i
                    valid_action_ids.append(action_id)
                    action_ids_ref_dict[i+4] = "switch to " + p.species.lower()
    else : 
    # Vérification des mouvements disponibles pour l'active Pokémon
        for i, move in enumerate(active.moves.values()):
            if move in battle.available_moves:
                valid_action_ids.append(i)
                action_ids_ref_dict[i] = ("move", move) 
        # Ajouter uniquement les Pokémon valides du banc (pas le Pokémon actif)
        for i,  p in enumerate(battle.available_switches):
            if not p.fainted:
                if 0 <= i <= 4:  # Banc de 2 à 6
                    action_id = 4 + i
                    valid_action_ids.append(action_id)
                    action_ids_ref_dict[i+4] = "switch to " + p.species.lower()

    # Création du masque
    action_mask = np.zeros(9, dtype=np.float32)

    # Filtrage : on ne prend que des indices entre 0 et 8 (valides)
    valid_action_ids = [action_id for action_id in valid_action_ids if 0 <= action_id < 9]

    # Remplir le masque avec les actions valides
    action_mask[valid_action_ids] = 1.0

    return valid_action_ids, action_mask, action_ids_ref_dict



import torch
import torch.nn.functional as F

def masked_dqn_loss(q_pred, target_q_values, action_mask):
    """
    Calcule la perte MSE uniquement sur les actions valides.
    
    Params:
    - q_pred : (batch_size, 9) → Q-values prédites par le modèle
    - target_q_values : (batch_size, 9) → cibles calculées (via Bellman)
    - action_mask : (batch_size, 9) → 1 si action valide, 0 sinon
    
    Returns:
    - loss : scalaire, moyenne de la perte MSE sur les actions valides uniquement
    """
    # On applique le masque pour ne garder que les Q-values valides
    mask = action_mask.bool()  # transforme en masque booléen

    # Calcul de la perte uniquement sur les entrées valides
    loss = F.mse_loss(q_pred[mask], target_q_values[mask])

    return loss


class MaskedDQNPolicy(DQNPolicy):
    def compute_q_loss(self, obs, actions, rewards, next_obs, dones, weights, target_net):
        # Q-values pour l’état actuel
        q_values = self.q_net(obs)  # (batch, 9)
        # Q-values pour l’état suivant
        with torch.no_grad():
            next_q_values = target_net(next_obs)
            next_q_values = next_q_values.max(dim=1)[0]  # max_a' Q(s', a')

        # Cible Bellman
        target = rewards + (1.0 - dones) * self.gamma * next_q_values

        # Q-value pour l’action prise
        q_values_taken = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)

        # 🎯 action_mask depuis les 9 premières colonnes de l’obs
        action_mask = obs[:, :9]  # (batch, 9)
        valid = torch.gather(action_mask, 1, actions.long().unsqueeze(1)).squeeze(1)  # (batch,)
        mask = valid > 0.5  # booléen : True si action valide

        # 🔹 MSE classique sur actions valides uniquement
        loss_valid = F.mse_loss(q_values_taken[mask], target[mask])

        # 🔸 Bonus : pénalité explicite pour actions invalides
        penalty_invalid = (1.0 - valid) * 1.0  # tu peux ajuster ce 1.0
        loss_penalty = penalty_invalid.mean()

        # 🎯 Total
        total_loss = loss_valid + loss_penalty
        return total_loss