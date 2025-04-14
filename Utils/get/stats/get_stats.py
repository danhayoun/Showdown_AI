#Fonctions pour calculer et renvoyer les stats
from poke_env.data import GenData
import numpy as np
import math
gen_data = GenData.from_gen(4)  # Gen 4

MAX_HP = 714 #Blissey max
STAT_MAX = 669.0 #Max stat other than hp

def get_stats(p) :
    v = np.zeros(6)
    base_stats = gen_data.pokedex[p.species]["baseStats"]

    hp = base_stats["hp"]
    order = ['atk', 'def', 'spa', 'spd', 'spe']
    static_stats = np.array([base_stats[k] for k in order], dtype=np.float32)

    hp = int(((2 * hp + 31 + (85 // 4)) * p.level) / 100 + p.level + 10) #Je fais confiance à p.level, j'espère il va pas me décevoir :O
    static_stats = np.floor(((2 * static_stats + 31 + (85 // 4)) * p.level) / 100 + 5)

    v[0] = hp/MAX_HP
    v[-5:] = static_stats/STAT_MAX

    return v 


