import random
import torch
import yaml
import os
import numpy as np
import logging

from environment.domino_env import DominoEnv, ACTION_DRAW, ACTION_PASS
from agents.ppo_lstm_agent import PpoLstmAgent
from environment.utils import DOMINO_TO_INDEX, INDEX_TO_DOMINO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_human_action(env: DominoEnv, obs_dict: dict) -> int:
    """Demande à l'utilisateur humain de choisir une action légale.

    Utilise env.get_legal_action_mask() et env._decode_action_code().
    """
    print("\n--- Votre Tour ---")
    legal_mask = env.get_legal_action_mask()
    legal_actions = []
    action_map = {} # Map choice index -> action_code

    for action_code, is_legal in enumerate(legal_mask):
        if is_legal:
            try:
                details = env._decode_action_code(action_code)
                action_type = details.get("type")
                description = "Action inconnue"

                if action_type == "play":
                    tile = details.get("tile")
                    end_val = details.get("end_value")
                    if end_val == -1:
                         description = f"Jouer {tile} (premier coup)"
                    else:
                         # Déterminer sur quelle extrémité logique (0 ou 1) l'action code correspond
                         target_end_idx = 0 if action_code < env._play_action_offset else 1
                         description = f"Jouer {tile} sur extrémité {target_end_idx} (valeur {end_val})"
                elif action_type == "draw":
                    description = "Piocher"
                elif action_type == "pass":
                    description = "Passer"

                legal_actions.append({"code": action_code, "desc": description})

            except Exception as e:
                 # Si _decode échoue pour une action légale, c'est un bug
                 logger.error(f"Erreur lors du décodage de l'action légale {action_code}: {e}")
                 legal_actions.append({"code": action_code, "desc": f"Action {action_code} (Erreur décodage)"})

    if not legal_actions:
        print("ERREUR: Aucune action légale trouvée pour l'humain via le masque!")
        # Tenter de passer par défaut en cas d'erreur?
        return ACTION_PASS

    # Trier pour un affichage cohérent (ex: play par tile, puis draw, puis pass)
    legal_actions.sort(key=lambda x: (x['code'] >= ACTION_DRAW, x['code']))

    print("Actions Légales:")
    for i, action_info in enumerate(legal_actions):
        choice_idx = i + 1
        print(f"{choice_idx}. {action_info['desc']}")
        action_map[choice_idx] = action_info['code']

    while True:
        try:
            choice = int(input(f"Entrez votre choix (1-{len(legal_actions)}): "))
            if 1 <= choice <= len(legal_actions):
                chosen_action_code = action_map[choice]
                return chosen_action_code
            else:
                print("Choix invalide.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre.")

def play_vs_agent(agent_path: str, config_path: str):
    """Lance une partie où un humain joue contre l'agent chargé."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env = DominoEnv(render_mode='human')

    action_dim = env.action_space.n

    agent = PpoLstmAgent(action_dim=action_dim, config=config)
    try:
        agent.load_model(agent_path)
        agent.network.eval()
        logger.info(f"Agent chargé depuis {agent_path}")
    except Exception as e:
        logger.error(f"Impossible de charger l'agent: {e}")
        return

    human_player_id = 0
    print(f"Vous êtes le Joueur {human_player_id}.")
    print("Lancement de la partie...")

    obs_dict, info = env.reset()
    agent.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        current_player = env.current_player
        env.render()
        legal_mask = env.get_legal_action_mask()

        if not np.any(legal_mask):
             # This case should ideally not happen if env logic is correct,
             # but added as a safeguard during reset/step debugging.
             if env.consecutive_passes < 2: # Check if it's a real unexpected state
                 logger.error("FATAL: Aucune action légale détectée par le masque! État:", info)
                 print("Erreur interne: Aucune action légale détectée. Forçage d'un passage.")
             # In a blocked state, passing might be the only theoretical option
             action = ACTION_PASS # Force pass if mask is empty
        else:
            action = -1 # Reset action
            if current_player == human_player_id:
                action = get_human_action(env, obs_dict)
            else:
                print("\n--- Tour de l'Agent ---")
                with torch.no_grad():
                    agent_action_code, _, _ = agent.select_action(obs_dict, info, legal_mask)
                    action = agent_action_code
                    # Afficher l'action de l'agent de manière plus informative
                    try:
                        agent_action_details = env._decode_action_code(action)
                        action_type = agent_action_details.get("type")
                        if action_type == "play":
                            tile = agent_action_details.get("tile")
                            end_val = agent_action_details.get("end_value")
                            target_end_idx = 0 if action < env._play_action_offset else 1
                            print(f"Agent joue: {tile} sur extrémité {target_end_idx} (valeur {end_val})")
                        elif action_type == "draw":
                            print("Agent pioche.")
                        elif action_type == "pass":
                            print("Agent passe.")
                        else:
                            print(f"Agent choisit action code {action} (type: {action_type})")
                    except Exception as e:
                        logger.error(f"Erreur décodage action agent {action}: {e}")
                        print(f"Agent choisit action code {action}")

        if action == -1:
            print("Erreur critique: Aucune action sélectionnée pour le tour.")
            break

        next_obs_dict, reward, terminated, truncated, info = env.step(action)
        obs_dict = next_obs_dict
        done = terminated or truncated

        if done:
            env.render() # Render final state after last action
            winner = env.winner
            print("\n--- Fin de la Partie ---")
            if winner == human_player_id: print("Félicitations, vous avez gagné !")
            elif winner == -1: print("Match Nul !")
            else: print("L'agent a gagné.")
            print("------------------------\n")
            break

    env.close()

def main():
    MODEL_TO_PLAY_VS = './models/ppo_domino_lstm_step563200.pth' 
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'ppo_config.yaml')

    if not os.path.exists(MODEL_TO_PLAY_VS):
         print(f"Erreur: Fichier modèle non trouvé {MODEL_TO_PLAY_VS}")
    elif not os.path.exists(CONFIG_FILE):
         print(f"Erreur: Fichier config non trouvé {CONFIG_FILE}")
    else:
        play_vs_agent(MODEL_TO_PLAY_VS, CONFIG_FILE)

if __name__ == "__main__":
    main()