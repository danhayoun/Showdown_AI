########### Les différents dictionnaires d'ids ###########
import numpy as np
import pandas as pd
from math import floor

##DF ##
df = pd.read_csv("data/csv/move_to_id.csv")  # ou le chemin local
moves_to_id = dict(zip(df["move"], df["id"]))
moves_to_id["hiddenpower"] = 186
for i in moves_to_id.keys() :
    moves_to_id[i] = moves_to_id[i] + 1

df = pd.read_csv("data/csv/species_to_id.csv")  # ou le chemin local
species_to_id = dict(zip(df["species"], df["id"]))
for i in species_to_id.keys() :
    species_to_id[i] = species_to_id[i] + 1

species_to_id["gastrodoneast"] = 100
species_to_id["gastrodonwest"] = 100

#Ajout de tous les types de unown
for c in "abcdefghijklmnopqrstuvwxyz":
    species_to_id[f"unown{c}"] = species_to_id["unown"]

species_to_id["unownquestion"] = species_to_id["unown"]
species_to_id["unownexclamation"] = species_to_id["unown"]



item_to_id = {
    "unknown_item": 0,
    None : 0,
    "None": 0,
    '': 0,
    "choiceband": 1,
    "choicescarf": 2,
    "choicespecs": 3,
    "expertbelt": 4,
    "focussash": 5,
    "leftovers": 6,
    "lifeorb": 7,
    "lumberry": 8,
    "powerherb": 9,
    "salacberry": 10,
    "sitrusberry": 11,
    "blacksludge": 12,
    "thickclub": 13,       # os de Ossatueur (Marowak)
    "lightball": 14,       # objet de Pikachu
    "flameplate": 15, 
    "fistplate": 16,
    "earthplate": 17, 
    "stoneplate": 18, 
    "zapplate": 19,
    "meadowplate": 20, 
    "icicleplate": 21, 
    "toxicplate": 22, 
    "mindplate": 23, 
    "insectplate": 24,
    "spookyplate": 25, 
    "dracoplate": 26, 
    "dreadplate": 27, 
    "ironplate": 28, 
    "skyplate": 29,
    "splashplate": 30,
    "damprock": 31,
    "blackglasses": 32,
    "silkscarf": 33,
    "toxicorb": 34,
    "griseousorb" : 35,
    "stick": 36,
    "lustrousorb" : 37,
    "souldew" : 38,
    "chestoberry": 39,
    "custapberry": 40
}

ability_to_id = {
    "adaptability": 1,
    "aftermath": 2,
    "airlock": 3,
    "anticipation": 4,
    "arenatrap": 5,
    "baddreams": 6,
    "battlearmor": 7,
    "blaze": 8,
    "chlorophyll": 9,
    "clearbody": 10,
    "cloudnine": 11,
    "colorchange": 12,
    "compoundeyes": 13,
    "cutecharm": 14,
    "download": 15,
    "drizzle": 16,
    "drought": 17,
    "dryskin": 18,
    "earlybird": 19,
    "filter": 20,
    "flamebody": 21,
    "flashfire": 22,
    "flowergift": 23,
    "forecast": 24,
    "forewarn": 25,
    "frisk": 26,
    "gluttony": 27,
    "guts": 28,
    "hugepower": 29,
    "hydration": 30,
    "hypercutter": 31,
    "immunity": 32,
    "innerfocus": 33,
    "insomnia": 34,
    "intimidate": 35,
    "ironfist": 36,
    "keeneye": 37,
    "leafguard": 38,
    "levitate": 39,
    "limber": 40,
    "liquidooze": 41,
    "magicguard": 42,
    "magnetpull": 43,
    "marvelscale": 44,
    "minus": 45,
    "moldbreaker": 46,
    "motordrive": 47,
    "multitype": 48,
    "naturalcure": 49,
    "noguard": 50,
    "overgrow": 51,
    "owntempo": 52,
    "pickup": 53,
    "plus": 54,
    "poisonheal": 55,
    "poisonpoint": 56,
    "pressure": 57,
    "purepower": 58,
    "quickfeet": 59,
    "rockhead": 60,
    "roughskin": 61,
    "runaway": 62,
    "sandstream": 63,
    "sandveil": 64,
    "scrappy": 65,
    "serenegrace": 66,
    "shadowtag": 67,
    "shedskin": 68,
    "shellarmor": 69,
    "shielddust": 70,
    "simple": 71,
    "skilllink": 72,
    "slowstart": 73,
    "sniper": 74,
    "snowcloak": 75,
    "snowwarning": 76,
    "solidrock": 77,
    "soundproof": 78,
    "speedboost": 79,
    "static": 80,
    "steadfast": 81,
    "stickyhold": 82,
    "sturdy": 83,
    "suctioncups": 84,
    "superluck": 85,
    "swarm": 86,
    "swiftswim": 87,
    "synchronize": 88,
    "tangledfeet": 89,
    "technician": 90,
    "thickfat": 91,
    "tintedlens": 92,
    "torrent": 93,
    "trace": 94,
    "truant": 95,
    "unburden": 96,
    "vitalspirit": 97,
    "voltabsorb": 98,
    "waterabsorb": 99,
    "waterveil": 100,
    "whitesmoke": 101,
    "wonderguard": 102
}

types = [
    "bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost",
    "grass", "ground", "ice", "normal", "poison", "psychic", "rock", "steel",
    "water"
]

# Générer les combinaisons mono- et duo-types, sans doublons (ordre alphabétique)
type_combos = set()

for t in types:
    type_combos.add((t,))
    for t2 in types:
        if t != t2:
            type_combos.add(tuple(sorted([t, t2])))

# Convertir en string clés
combo_keys = ["_".join(combo) for combo in sorted(type_combos)]
combo_keys.append("none")

# Générer le dictionnaire
types_to_id = {k: i+1 for i, k in enumerate(combo_keys)}



from poke_env.environment.weather import Weather
from poke_env.environment.side_condition import SideCondition
weather_to_id = {
    None: 0,  # Pour les cas où il n'y a pas de météo
    Weather.UNKNOWN: 0,
    Weather.SUNNYDAY: 1,
    Weather.RAINDANCE: 2,
    Weather.SANDSTORM: 3,
    Weather.HAIL: 4
}
side_condition_to_id = {
    SideCondition.STEALTH_ROCK: 0,
    SideCondition.SPIKES: 1,
    SideCondition.TOXIC_SPIKES: 3,
    SideCondition.REFLECT: 5,
    SideCondition.LIGHT_SCREEN: 7,
    SideCondition.TAILWIND: 9,
    SideCondition.STICKY_WEB: 11  # (pas présent en Gen 4, mais au cas où tu fais d'autres formats)
}

import json
from poke_env.environment.pokemon import Pokemon

# 1. Charger le fichier
with open("json/sets.json", "r") as f:
    sets_data = json.load(f)

# 2. Préremplir le cache
POKEMON_CACHE = {}
for species in sets_data.keys():
    try:
        POKEMON_CACHE[species.lower()] = Pokemon(gen=4, species=species)
    except Exception as e:
        print(f"⚠️ Erreur pour {species} : {e}")


# Ajouter explicitement toutes les formes d'Unown
for c in "abcdefghijklmnopqrstuvwxyz":
    POKEMON_CACHE[f"unown{c}"] = Pokemon(gen=4, species="unown")
POKEMON_CACHE["unownquestion"] = Pokemon(gen=4, species="unown")
POKEMON_CACHE["unownexclamation"] = Pokemon(gen=4, species="unown")

# Ajouter explicitement les deux formes de Gastrodon
POKEMON_CACHE["gastrodoneast"] = Pokemon(gen=4, species="gastrodoneast")



### Functions ###

def is_my_pokemon_locked(dict_observation) :
    poke = dict_observation.get("active_pokemon", None)
    if poke is None:
        return 0  # Pas de pokémon actif, donc pas locké
    item = getattr(poke, "item",None)
    is_choice = item in {"choiceband", "choicescarf", "choicespecs"}

    is_encored = False
    if hasattr(poke, "volatiles") and "encore" in getattr(poke, "volatiles", {}):
        is_encored = True
    elif hasattr(poke, "encore_turns") and getattr(poke, "encore_turns", 0) > 0:
        is_encored = True

    return int(is_choice or is_encored)

def get_my_switches_with_all_infos(switches,dict) : 

    # Dictionnaire contenant les ObservedPokemon (via battle.current_observation.__dict__["team"])
    team_obs_dict = dict["team"]

    enriched_switches = []

    for poke in switches:
        # Trouver l'ObservedPokemon correspondant en comparant le species
        matched_obs = next(
            (obs for obs in team_obs_dict.values()
             if hasattr(obs, "species") and obs.species == poke.species),
            None
        )

        if matched_obs:
            enriched_switches += ObservedPokemon_to_list(matched_obs,is_active_pokemon=0,is_opponent_team=0)
        else:
            # Fallback si non trouvé
            enriched_switches += ["UNKNOWN", 0.0, "NONE"] + [0]*7 + [0]*6

    return enriched_switches
    
def get_opponent_team_with_all_infos(dict,battle) : 

    # Dictionnaire contenant les ObservedPokemon (via battle.current_observation.__dict__["team"])
    team_obs_dict = dict["opponent_team"]
    #enlever le pokemon actif adverse
    active = battle.opponent_active_pokemon

    opponent_team_list = []
    for poke in team_obs_dict.values() :
        if poke.species != active.species:
            opponent_team_list += ObservedPokemon_to_list(poke,is_active_pokemon=0,is_opponent_team=1)


    return opponent_team_list

def pad_list(liste, max_len, pad_value=0.0):
    """Retourne une liste de longueur max_len,
    paddée avec pad_value si besoin."""
    if len(liste) > max_len :
        print(f"[WARN] La liste d'entrée est plus longue ({len(liste)}) que max_len ({max_len}) — elle va être tronquée.")
    res = list(liste[:max_len])
    while len(res) < max_len:
        res.append(pad_value)
    return res

def pad_vector(vector, max_len, pad_value=0.0, dtype=np.float32):
    """Retourne un np.array de longueur max_len,
    paddé avec pad_value si besoin."""
    arr = np.full(max_len, pad_value, dtype=dtype)
    arr[:min(len(vector), max_len)] = vector[:max_len]
    return arr

def get_weather(dict_observation) :
    obs_weather = dict_observation["weather"]
    if obs_weather:
        weather_enum, turns_left = next(iter(obs_weather.items()))
    else:
        weather_enum, turns_left = None, 0  # ou None, 0 selon ton mapping

    weather = [weather_enum, turns_left]
    return weather

from poke_env.environment.side_condition import SideCondition
from Players.Player5 import side_condition_to_id
#Fonction pour obtenir les sides conditions
#Remarque : au lieu d'appeller ça conditons on pourrait peut-être appeller ça hazard...
MAX_LEN_SIDE_CONDITIONS = 12
def get_side_conditions(dict_observation,is_my_side) :
    vector = [0 for i in range(MAX_LEN_SIDE_CONDITIONS)] #padding déjà fait du coup
    if is_my_side :
        obs_conditions = dict_observation["side_conditions"]
    else : 
        obs_conditions = dict_observation["opponent_side_conditions"]
    
    if obs_conditions :
        for i in obs_conditions.keys() :
            if i== SideCondition.STEALTH_ROCK or i== SideCondition.STICKY_WEB :
                id = side_condition_to_id[i]
                vector[id] = 1
            else :
                id = side_condition_to_id[i]
                vector[id] = 1
                vector[id+1] = obs_conditions[i] #On prend la valeur associée au hazard dans le dico

    #On normalise car ce vecteur rentre directement dans le réseau profond sans passer par l'embedding
    vector[2] = vector[2]/3 # 3 spikes maximum
    vector[4] = vector[4]/2 # 2 t spikes maximum
    vector[6] = vector[6]/8 # 8 tours de screen max
    vector[8] = vector[8]/8 # 8 tours de screen max
    vector[10] = vector[10]/4 # 4 tours de tailwind max

    return vector




def ObservedPokemon_to_list(pokemon,is_active_pokemon,is_opponent_team) -> list: 
    # Nom du Pokémon
    species = getattr(pokemon, "species", "UNKNOWN")
    species_id = species_to_id[species]
    #Norm
    species_id_norm = np.float32(species_id/len(species_to_id))

    item = getattr(pokemon, "item", None)

    item_id = item_to_id[item]
    item_id_norm = np.float32(item_id/len(item_to_id))

    if species.lower() in POKEMON_CACHE:
        p = POKEMON_CACHE[species.lower()]
    else:
        print(f"[WARN] Espèce inconnue dans le cache : {species}")
        p = Pokemon(gen=4, species=species)  # Fallback (devrait jamais arriver si cache bien rempli)


    # Fraction de vie (float entre 0 et 1)
    hp = getattr(pokemon, "current_hp_fraction", 0.0) or 0.0

    # Types (liste de strings ou [])
    types = [p.type_1.name.lower() if p.type_1 else None, p.type_2.name.lower() if p.type_2 else None]
    if p.type_1 and p.type_2 :
        types = sorted([t for t in types])
    type_str = types[0] if types[1] is None else f"{types[0]}_{types[1]}"
    type_id = types_to_id[type_str]
    type_id_norm = np.float32(type_id/len(types_to_id))

    # Statut (string ou "NONE")
    status = getattr(pokemon, "status", None)
    status_str = status.name if status else "NONE"
    status_to_id = {
     'NONE': 0,
    'PAR': 1,     # paralysie
    'SLP': 2,     # sommeil
    'BRN': 3,     # brûlure
    'FRZ': 4,     # gel
    'PSN': 5,     # poison
    'TOX': 6,     # poison grave
    'FNT': 7
        }
    status_id = status_to_id[status_str] 
    status_id_norm = status_id/len(status_to_id)

    #Boosts
    if(is_active_pokemon == 1) :

        # Boosts (dict ou [0]*7)
        boosts = getattr(pokemon, "boosts", None)
        boosts_list = list(boosts.values()) if boosts else [0] * 7
        boosts_list_norm = [np.float32(b / 8) for b in boosts_list]

    # Stats (dict ou [0]*6) noramlisées
    if is_opponent_team == 0 : #Si c'est pas l'équipe adverse :

        stats = getattr(pokemon, "stats", None)
        stat_list = list(stats.values()) if stats else [0] * 6
        stats_list = [np.float32(x / 504) if x is not None else 0.0 for x in stat_list]
        #print(f"DEBUUUUUUUUG VALEUR DE STATS_LIST: {stats_list}")

    else : #si c'est l'équipe adverse, on fait autrement
        level = p.level if hasattr(p, "level") and p.level else 80
        hp_stat = floor(((2 * p.base_stats["hp"] + 31 + 84 // 4) * level) / 100) + level + 10
        if hp == None : 
            hp = 1.0
            #On rajoute un quasi-normalize, pour aller autour de 1 sans que ça soit trop petit, donc on prend pas l'exemple leuphorie 714 stats en PV avec lvl 100, nature, EVs
            
        atk = floor((2 * p.base_stats["atk"] + 31 + 84 // 4) * level / 100 + 5) 
        def_ = floor((2 * p.base_stats["def"] + 31 + 84 // 4) * level / 100 + 5) 
        spa = floor((2 * p.base_stats["spa"] + 31 + 84 // 4) * level / 100 + 5) 
        spd = floor((2 * p.base_stats["spd"] + 31 + 84 // 4) * level / 100 + 5) 
        spe = floor((2 * p.base_stats["spe"] + 31 + 84 // 4) * level / 100 + 5) 
        if atk is None :
            atk = 0
        if def_ is None : 
            def_ = 0
        if spa is None : 
            spa = 0
        if spd is None : 
            spd = 0 
        if spe is None : 
            spe = 0
        stats_list = [np.float32(hp_stat/504),np.float32(atk/504),np.float32(def_/504),np.float32(spa/504),np.float32(spd/504),np.float32(spe/504)]
        #print(f"DEBUUUUUUUUG VALEUR DE STATS_LIST: {stats_list}")
    #moovepool
    moves_dict = getattr(pokemon, "moves", None)
    move_names = list(moves_dict.keys()) if moves_dict else []

    # Pour avoir toujours 4 valeurs :
    while len(move_names) < 4:
        move_names.append("None")

    move_names_id = [
        moves_to_id[x.lower().replace(" ", "")] if x and x.lower() != "none" else 0 #Padding 0
        for x in move_names
        ]
    normalized_moves = [np.float32(m / len(moves_to_id)) for m in move_names_id[:4]]
        
    ability = p.ability
    ability_id = ability_to_id[ability] if ability else 0 # Padding 0 
    ability_id_norm = np.float32(ability_id/len(ability_to_id))
        
        
    if(is_active_pokemon == 1) :
        #on normalise
        return [
            species_id_norm,
            item_id_norm,
            hp,
            ability_id_norm,
            type_id_norm,
            status_id_norm,
            *boosts_list_norm,
            *stats_list,
            *normalized_moves
        ]
    else : 
        return [
            species_id_norm,
            item_id_norm,
            hp,
            ability_id_norm,
            type_id_norm,
            status_id_norm,
            *stats_list,
            *normalized_moves
        ]
    

