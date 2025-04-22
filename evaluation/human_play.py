# evaluation/human_play.py
import random
import torch
import yaml
import os
import numpy as np
import logging

from environment.domino_env import DominoEnv
from agents.ppo_lstm_agent import PpoLstmAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_human_action(env: DominoEnv, obs_dict: dict, legal_actions_map: dict) -> int:
    """Demande à l'utilisateur humain de choisir une action."""
    print("\n--- Votre Tour ---")
    print("Actions Légales:")
    action_choices = {}
    choice_idx = 1
    pass_code = -1
    draw_code = -1

    playable_moves_display = []
    for code, details in legal_actions_map.items():
         if details['type'] == 'play':
              try:
                   tile = env.player_hands[env.current_player][details['tile_idx']]
                   playable_moves_display.append((code, details, tile))
              except IndexError:
                   continue
         elif details['type'] == 'draw':
             draw_code = code
         elif details['type'] == 'pass':
             pass_code = code

    playable_moves_display.sort(key=lambda x: x[2])

    for code, details, tile in playable_moves_display:
         end_str = f" sur {details['end']}" if details['end'] != -1 else " (premier coup)"
         action_str = f"Jouer {tile}{end_str}"
         print(f"{choice_idx}. {action_str}")
         action_choices[choice_idx] = code
         choice_idx += 1

    if draw_code != -1:
        print(f"{choice_idx}. Piocher")
        action_choices[choice_idx] = draw_code
        choice_idx += 1
    if pass_code != -1:
        print(f"{choice_idx}. Passer")
        action_choices[choice_idx] = pass_code
        choice_idx += 1

    while True:
        try:
            choice = int(input(f"Entrez votre choix (1-{choice_idx - 1}): "))
            if 1 <= choice < choice_idx:
                return action_choices[choice]
            else:
                print("Choix invalide.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre.")

def play_vs_agent(agent_path: str, config_path: str):
    """Lance une partie où un humain joue contre l'agent chargé."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env = DominoEnv(render_mode='human') 

    obs_sample = env.observation_space.sample()
    obs_dim_approx = sum(np.prod(v.shape) if hasattr(v, 'shape') else 1 for k, v in obs_sample.items() if k != 'board_sequence')
    obs_dim_approx = int(obs_dim_approx)  
    action_dim = env.action_space.n

    agent = PpoLstmAgent(obs_dim_approx, action_dim, config)
    try:
        agent.load_model(agent_path)
        agent.network.eval()
        logger.info(f"Agent chargé depuis {agent_path}")
    except Exception as e:
        logger.error(f"Impossible de charger l'agent: {e}")
        return

    human_player_id = random.choice([0, 1])
    # human_player_id = 0 
    print(f"Vous êtes le Joueur {human_player_id}.")
    print("Lancement de la partie...")

    obs_dict, info = env.reset()
    agent.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        current_player = env.current_player
        env.render() 
        legal_mask = env.get_legal_action_mask()
        legal_actions_map = env._get_legal_actions_encoded() # Pour l'affichage humain

        if not np.any(legal_mask):
             print("Erreur: Aucune action légale détectée par le masque. Le jeu devrait être terminé?")
             break

        action = -1
        if current_player == human_player_id:
            action = get_human_action(env, obs_dict, legal_actions_map)
        else:
            print("\n--- Tour de l'Agent ---")
            with torch.no_grad():
                action, _, _ = agent.select_action(obs_dict, legal_mask)
                action_details = legal_actions_map.get(action)
                if action_details:
                     if action_details['type'] == 'play':
                           try:
                                tile = env.player_hands[current_player][action_details['tile_idx']]
                                end_str = f" sur {action_details['end']}" if action_details['end'] != -1 else ""
                                print(f"Agent joue: {tile}{end_str}")
                           except: print(f"Agent joue (action code {action})") # Fallback
                     elif action_details['type'] == 'draw': print("Agent pioche.")
                     elif action_details['type'] == 'pass': print("Agent passe.")
                     else: print(f"Agent choisit l'action code {action}")
                else: print(f"Agent choisit l'action code {action}") # Si non trouvé dans map


        if action != -1:
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            obs_dict = next_obs_dict
            done = terminated or truncated
            if done:
                 env.render() 
                 winner = env.winner
                 print("\n--- Fin de la Partie ---")
                 if winner == human_player_id: print("Félicitations, vous avez gagné !")
                 elif winner == -1: print("Match Nul !")
                 else: print("L'agent a gagné.")
                 print("------------------------\n")
                 break
        else:
             print("Erreur dans la sélection d'action.")
             break

    env.close()

def main():
    MODEL_TO_PLAY_VS = './models/ppo_domino_lstm_step102400.pth' # Modèle entraîné
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'ppo_config.yaml')

    if not os.path.exists(MODEL_TO_PLAY_VS):
         print(f"Erreur: Fichier modèle non trouvé {MODEL_TO_PLAY_VS}")
    elif not os.path.exists(CONFIG_FILE):
         print(f"Erreur: Fichier config non trouvé {CONFIG_FILE}")
    else:
        play_vs_agent(MODEL_TO_PLAY_VS, CONFIG_FILE)

if __name__ == "__main__":
    main()