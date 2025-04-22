import random
from typing import List, Tuple, Optional, Dict, Any
import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import logging

from .domino_tile import DominoTile
from .utils import MAX_DOMINO_VALUE, HAND_SIZE, TOTAL_TILES, generate_all_dominos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DominoEnv(gym.Env):
    """
    Environnement Gymnasium pour le jeu de Domino avec pioche.

    Action Space:
    - Jouer un domino: (index_tuile_main, end_val) -> nécessite un encodage/décodage
    - Piocher: Action spéciale si aucune tuile jouable ET pioche non vide.
    - Passer: Action spéciale si aucune tuile jouable ET pioche vide.

    Observation Space: (Exemple, à adapter pour PPO/LSTM)
    - Main du joueur (encodée)
    - Plateau (extrémités ouvertes, peut-être dominos joués)
    - Taille de la main de l'adversaire
    - Taille de la pioche
    - Indicateur de qui joue
    - (Optionnel: Sortie du Belief Network)
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.verbose = render_mode == 'human'
        self.render_mode = render_mode

        obs_components = {
            'my_hand': spaces.MultiBinary(len(generate_all_dominos())), 
            'open_ends': spaces.MultiDiscrete(np.array([MAX_DOMINO_VALUE + 2] * 2)),
            'board_sequence': spaces.Sequence(spaces.Discrete(len(generate_all_dominos()))), 
            'opponent_hand_size': spaces.Discrete(HAND_SIZE + 1),
            'draw_pile_size': spaces.Discrete(TOTAL_TILES - 2 * HAND_SIZE + 1),
            'current_player': spaces.Discrete(2),
        }
        self.observation_space = spaces.Dict(obs_components)

        max_possible_play_actions = HAND_SIZE * 2 
        self.action_space = spaces.Discrete(max_possible_play_actions + 2) 
        self._action_draw = max_possible_play_actions
        self._action_pass = max_possible_play_actions + 1


        self.all_tiles = generate_all_dominos()

    def _get_obs(self) -> Dict[str, Any]:
        """Retourne l'observation actuelle pour l'agent."""
        # --- Encodage de l'Observation (à complexifier pour PPO/LSTM) ---
        my_hand_encoding = np.zeros(len(self.all_tiles), dtype=np.int8)
        for tile in self.player_hands[self.current_player]:
             # Trouver l'index global du domino
             try:
                idx = self.all_tiles.index(tile)
                my_hand_encoding[idx] = 1
             except ValueError: # Ne devrait pas arriver si all_tiles est correct
                 logger.error(f"Tuile {tile} de la main non trouvée dans all_tiles!")
                 pass # Gérer l'erreur comme approprié

        # Extrémités: utiliser une valeur spéciale (ex: MAX+1) si < 2 extrémités
        ends = sorted(self.open_ends) # Tri pour la cohérence
        if len(ends) == 0:
            open_ends_encoding = np.array([MAX_DOMINO_VALUE + 1, MAX_DOMINO_VALUE + 1])
        elif len(ends) == 1:
            open_ends_encoding = np.array([ends[0], MAX_DOMINO_VALUE + 1])
        else:
             open_ends_encoding = np.array([ends[0], ends[1]])

        board_sequence_encoding = [self.all_tiles.index(t) for t in self.board if t in self.all_tiles]

        return {
            'my_hand': my_hand_encoding,
            'open_ends': open_ends_encoding,
            'board_sequence': board_sequence_encoding, # Note: spaces.Sequence peut nécessiter un traitement spécial
            'opponent_hand_size': np.array([len(self.player_hands[1 - self.current_player])]),
            'draw_pile_size': np.array([len(self.draw_pile)]),
            'current_player': np.array([self.current_player]),
        }

    def _get_info(self) -> Dict[str, Any]:
        """Retourne des informations de débogage/diagnostique."""
        return {
            "player_0_hand": self.player_hands[0],
            "player_1_hand": self.player_hands[1],
            "board": self.board,
            "open_ends": self.open_ends,
            "draw_pile_size": len(self.draw_pile),
            "consecutive_passes": self.consecutive_passes,
            "scores": self.scores,
            "legal_actions": self._get_legal_actions_encoded() # Peut être utile pour le masquage
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Réinitialise le jeu."""
        super().reset(seed=seed) # Important pour gérer le générateur de nombres aléatoires
        random.seed(seed) # Assurer la reproductibilité si seed est fourni

        logger.info("Réinitialisation de la partie de domino")
        shuffled_tiles = self.all_tiles[:]
        random.shuffle(shuffled_tiles)

        self.player_hands = [shuffled_tiles[:HAND_SIZE], shuffled_tiles[HAND_SIZE:2*HAND_SIZE]]
        self.draw_pile = shuffled_tiles[2*HAND_SIZE:] 

        self.board = []
        self.open_ends = []
        self.current_player = self._determine_first_player()
        self.scores = [0, 0]
        self.history = []
        self.done = False
        self.winner = -1
        self.consecutive_passes = 0
        self.last_action_description = "Game Start"

        if self.verbose:
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _determine_first_player(self) -> int:
        """Détermine qui commence (double le plus élevé, sinon domino le plus fort)."""
        highest_double = -1
        first_player = -1
        start_tile_p0 = None
        start_tile_p1 = None

        for player_idx, hand in enumerate(self.player_hands):
            for tile in hand:
                if tile.is_double() and tile.side1 > highest_double:
                    highest_double = tile.side1
                    first_player = player_idx
                    if player_idx == 0: start_tile_p0 = tile
                    else: start_tile_p1 = tile

        if first_player == -1:
            highest_value = -1
            for player_idx, hand in enumerate(self.player_hands):
                 # hand.sort(key=lambda t: (t.get_value(), max(t.get_sides())), reverse=True)
                for tile in hand:
                    if tile.get_value() > highest_value:
                        highest_value = tile.get_value()
                        first_player = player_idx
                        if player_idx == 0: start_tile_p0 = tile
                        else: start_tile_p1 = tile
                    elif tile.get_value() == highest_value:
                        current_highest_side = max(start_tile_p0.get_sides()) if first_player == 0 else max(start_tile_p1.get_sides())
                        if max(tile.get_sides()) > current_highest_side:
                            first_player = player_idx
                            if player_idx == 0: start_tile_p0 = tile
                            else: start_tile_p1 = tile


        if first_player == -1:
            first_player = random.choice([0, 1])

        logger.info(f"Le joueur {first_player} commence la partie.")
        return first_player

    def _get_playable_moves(self, player_idx: Optional[int] = None) -> List[Tuple[int, int, bool]]:
        """
        Retourne les coups jouables pour le joueur.
        Format: (index_tuile_main, valeur_extremite_a_jouer, doit_retourner_tuile)
        Si le plateau est vide, valeur_extremite_a_jouer est None (ou une valeur spéciale).
        """
        if player_idx is None:
            player_idx = self.current_player

        hand = self.player_hands[player_idx]
        playable_moves = []

        if not self.board:
            for i, tile in enumerate(hand):
                playable_moves.append((i, -1, False))
        else:
            ends_to_check = list(set(self.open_ends))
            for i, tile in enumerate(hand):
                s1, s2 = tile.get_sides()
                for end in ends_to_check:
                    if s1 == end:
                        playable_moves.append((i, end, False))
                    elif s2 == end:
                        playable_moves.append((i, end, True)) 


        unique_plays = {}
        for idx, end, flip in playable_moves:
             key = (idx, end)
             if key not in unique_plays:
                 unique_plays[key] = (idx, end, flip)

        return list(unique_plays.values())


    def _get_legal_actions_encoded(self) -> Dict[int, Any]:
        """
        Retourne un dictionnaire des actions légales encodées pour l'agent.
        Clé: Index d'action dans l'espace discret.
        Valeur: Description/paramètres de l'action (utile pour le débogage ou l'agent).
        """
        legal_actions = {}
        playable_moves = self._get_playable_moves()

        if playable_moves:
            action_idx_counter = 0
            for tile_hand_idx, end_to_match, flip in playable_moves:
                current_action_encoded = action_idx_counter # A REVOIR SERIEUSEMENT
                if current_action_encoded < self._action_draw: # Assurer qu'on ne dépasse pas
                     legal_actions[current_action_encoded] = {"type": "play", "tile_idx": tile_hand_idx, "end": end_to_match, "flip": flip}
                     action_idx_counter += 1
                else:
                      logger.warning("Dépassement de l'encodage d'action de jeu !")
                      break 

        elif self.draw_pile:
            legal_actions[self._action_draw] = {"type": "draw"}
        else: 
            legal_actions[self._action_pass] = {"type": "pass"}

        return legal_actions

    def _decode_action(self, action_code: int) -> Dict[str, Any]:
        """ Traduit un code d'action de l'espace discret en action logique. """
        legal_actions_map = self._get_legal_actions_encoded()
        if action_code in legal_actions_map:
            return legal_actions_map[action_code]
        else:
           
           
            logger.warning(f"Action illégale {action_code} reçue ! Actions légales: {list(legal_actions_map.keys())}")
            
            if legal_actions_map:
                fallback_action_code = list(legal_actions_map.keys())[0]
                logger.warning(f"Action remplacée par défaut: {fallback_action_code}")
                return legal_actions_map[fallback_action_code]
            else:      
                 logger.error("Aucune action légale trouvée, même pas piocher ou passer !")
                 return {"type": "pass"}


    def step(self, action_code: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """ Exécute une étape du jeu. """
        if self.done:
            logger.warning("step() appelé sur un environnement terminé.")
            return self._get_obs(), 0.0, True, False, self._get_info()


        decoded_action = self._decode_action(action_code)
        action_type = decoded_action["type"]
        player = self.current_player
        reward = 0.0
        terminated = False
        truncated = False 

        self.last_action_description = f"Player {player}: {action_type}"

        if action_type == "play":
            tile_idx = decoded_action["tile_idx"]
            end = decoded_action["end"]
            flip = decoded_action["flip"] 

            hand = self.player_hands[player]

            if tile_idx < 0 or tile_idx >= len(hand):
                 logger.error(f"Action Jouer invalide: indice de tuile {tile_idx} hors limites pour main {hand}")
                 action_type = "pass"
            else:
                tile_to_play = hand.pop(tile_idx)
                self.last_action_description += f" tile {tile_to_play} matching {end}"

                if not self.board: 
                    self.board.append(tile_to_play)
                    s1, s2 = tile_to_play.get_sides()
                    self.open_ends = [s1, s2]

                else:
                    s1, s2 = tile_to_play.get_sides()
                    connected_side = s1 if s1 == end else s2
                    new_open_side = s2 if s1 == end else s1

                   
                    if self.open_ends[0] == connected_side:
                        self.open_ends[0] = new_open_side
                        self.board.insert(0, tile_to_play) 
                    elif self.open_ends[1] == connected_side:
                        self.open_ends[1] = new_open_side
                        self.board.append(tile_to_play) 
                    else:
                        logger.error(f"Erreur logique: l'extrémité {connected_side} de la tuile {tile_to_play} ne correspond à aucune extrémité ouverte {self.open_ends}")
                        hand.insert(tile_idx, tile_to_play)
                        action_type = "pass" 

                if action_type == "play": # Si le coup a réussi
                    self.history.append((player, tile_to_play, end))
                    self.consecutive_passes = 0

                    if not hand:
                        terminated = True
                        self.winner = player
                        reward = self._calculate_score(player) 
                        self.last_action_description += " -> WIN!"
                        logger.info(f"Joueur {player} a gagné!")

        elif action_type == "draw":
             if not self.draw_pile:
                 logger.error("Action Piocher invalide: la pioche est vide.")
                 action_type = "pass" 
             else:
                drawn_tile = self.draw_pile.pop(0)
                self.player_hands[player].append(drawn_tile)
                self.history.append((player, "draw", None)) 
                self.consecutive_passes = 0
                self.last_action_description += f" -> Draws {drawn_tile}. Hand size: {len(self.player_hands[player])}"
                logger.info(f"Joueur {player} pioche {drawn_tile}")
        if action_type == "pass":
            can_play = bool(self._get_playable_moves(player))
            if self.draw_pile and not can_play:
                 logger.error(f"Joueur {player} a tenté de passer alors qu'il DOIT piocher.")
                 drawn_tile = self.draw_pile.pop(0)
                 self.player_hands[player].append(drawn_tile)
                 self.history.append((player, "forced_draw", None))
                 self.consecutive_passes = 0
                 self.last_action_description = f"Player {player}: Forced Draw (illegal pass) -> Draws {drawn_tile}"
            elif can_play:
                logger.error(f"Joueur {player} a tenté de passer alors qu'il pouvait jouer.")
                self.history.append((player, "pass_illegal", None))
                self.consecutive_passes += 1
                self.last_action_description = f"Player {player}: Illegal Pass (could play)"
                self.current_player = 1 - player
            else: 
                self.history.append((player, "pass", None))
                self.consecutive_passes += 1
                self.last_action_description = f"Player {player}: Pass (legal)"
                logger.info(f"Joueur {player} passe son tour (légal)")
                if self.consecutive_passes >= 2:
                    terminated = True
                    reward = self._calculate_score_blocked() 
                    self.last_action_description += " -> BLOCKED GAME!"
                    logger.info(f"Jeu bloqué! Scores de main: P0={self._get_hand_value(0)}, P1={self._get_hand_value(1)}. Winner: {self.winner}")
                else:
                    self.current_player = 1 - player


        if action_type == "play" or (action_type == "pass" and self.consecutive_passes < 2):
             if not terminated: 
                 if not (action_type == "draw" or decoded_action.get("forced_draw")):
                      self.current_player = 1 - player


        self.done = terminated or truncated 
        if self.verbose:
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        final_reward_for_acting_player = 0.0
        if terminated:
            if self.winner == player:
                final_reward_for_acting_player = 1.0
            elif self.winner == -1: 
                final_reward_for_acting_player = 0.0
            else: 
                final_reward_for_acting_player = -1.0


        return observation, final_reward_for_acting_player, terminated, truncated, info

    def _get_hand_value(self, player_idx: int) -> int:
        """Calcule la somme des points dans la main d'un joueur."""
        return sum(tile.get_value() for tile in self.player_hands[player_idx])

    def _calculate_score(self, winner: int) -> float:
        """Calcule le score (récompense) quand un joueur gagne en vidant sa main."""
        if winner == -1: return 0.0 
        opponent = 1 - winner
        return 1.0 

    def _calculate_score_blocked(self) -> float:
        """Détermine le gagnant et la récompense quand le jeu est bloqué."""
        score0 = self._get_hand_value(0)
        score1 = self._get_hand_value(1)

        if score0 < score1:
            self.winner = 0
            # self.scores[0] += score1 # Si on compte les points
            reward_player0 = 1.0
            reward_player1 = -1.0
        elif score1 < score0:
            self.winner = 1
            # self.scores[1] += score0
            reward_player0 = -1.0
            reward_player1 = 1.0
        else: 
            self.winner = -1 # Nul
            reward_player0 = 0.0
            reward_player1 = 0.0

        last_player = 1 - self.current_player # Celui qui a joué en dernier (passé)
        return reward_player0 if last_player == 0 else reward_player1


    def render(self):
        if self.render_mode is None:
            # gym.logger.warn("You are calling render method without specifying any render mode.")
            return
        if self.render_mode == "human":
            print("\n" + "="*50)
            print(f"Dernière Action: {self.last_action_description}")
            print(f"Tour du Joueur: {self.current_player}")
            print(f"Plateau: {self.board}")
            print(f"Extrémités Ouvertes: {self.open_ends}")
            print(f"Main Joueur 0 ({len(self.player_hands[0])} tuiles): {sorted(self.player_hands[0])}")
            print(f"Main Joueur 1 ({len(self.player_hands[1])} tuiles): {sorted(self.player_hands[1])}")
            print(f"Pioche: {len(self.draw_pile)} tuiles")
            print(f"Passes Consécutives (après pioche vide): {self.consecutive_passes}")
            # print(f"Scores: {self.scores}")
            if self.done:
                print("-" * 20)
                print(f"Partie Terminée! Gagnant: {'Joueur ' + str(self.winner) if self.winner != -1 else 'Match Nul'}")
                print(f"Score Main Joueur 0: {self._get_hand_value(0)}")
                print(f"Score Main Joueur 1: {self._get_hand_value(1)}")
            print("="*50 + "\n")
        # elif self.render_mode == "rgb_array":
            # return self._render_frame()
        else:
             raise ValueError(f"Mode de rendu non supporté : {self.render_mode}")


    def close(self):
        """Nettoyage à la fermeture de l'environnement."""
        pass

    def get_legal_action_mask(self) -> np.ndarray:
        """
        Retourne un masque booléen/binaire pour les actions légales.
        Très important pour les algorithmes comme PPO/DQN.
        La taille doit correspondre à self.action_space.n
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        legal_actions_map = self._get_legal_actions_encoded()
        for action_code in legal_actions_map.keys():
            if 0 <= action_code < self.action_space.n:
                 mask[action_code] = True
            else:
                 logger.error(f"Code d'action légale {action_code} hors des limites de l'espace d'action {self.action_space.n}")
        return mask