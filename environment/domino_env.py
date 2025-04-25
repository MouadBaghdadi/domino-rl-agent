import random
from typing import List, Tuple, Optional, Dict, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

from .domino_tile import DominoTile
from .utils import MAX_DOMINO_VALUE, HAND_SIZE, TOTAL_TILES, generate_all_dominos, ALL_DOMINOS, DOMINO_TO_INDEX, INDEX_TO_DOMINO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_TILE_TYPES = len(ALL_DOMINOS) 
PLAY_ACTION_OFFSET = NUM_TILE_TYPES
# Action codes:
# 0 to 27: Play tile i on open_ends[0] (or first move)
# 28 to 55: Play tile i on open_ends[1]
# 56: Draw
# 57: Pass
ACTION_PLAY_END_0_START = 0
ACTION_PLAY_END_0_END = NUM_TILE_TYPES 
ACTION_PLAY_END_1_START = PLAY_ACTION_OFFSET 
ACTION_PLAY_END_1_END = PLAY_ACTION_OFFSET + NUM_TILE_TYPES 
ACTION_DRAW = PLAY_ACTION_OFFSET * 2
ACTION_PASS = ACTION_DRAW + 1      
TOTAL_ACTIONS = ACTION_PASS + 1    

class DominoEnv(gym.Env):
    """
    Environnement Gymnasium pour le jeu de Domino avec pioche (Double-Six).

    Action Space (Discrete(58)):
    - Indices 0 à 27: Jouer la tuile TILE_i sur l'extrémité ouverte 0 (si possible).
    - Indices 28 à 55: Jouer la tuile TILE_(i-28) sur l'extrémité ouverte 1 (si possible).
    - Index 56: Piocher ('draw'). Légal si aucun coup 'play' possible ET pioche non vide.
    - Index 57: Passer ('pass'). Légal si aucun coup 'play' possible ET pioche vide.

    Observation Space (Dict):
    - my_hand (MultiBinary): Indique quels dominos (parmi ALL_DOMINOS) sont dans la main.
    - open_ends (Box): Les deux valeurs numériques aux extrémités (-1 si pas d'extrémité).
    - board_value_counts (Box): Nombre de fois où chaque valeur (0-6) apparaît sur le plateau.
    - opponent_hand_size (Discrete): Nombre de tuiles chez l'adversaire.
    - draw_pile_size (Discrete): Nombre de tuiles dans la pioche.
    - current_player (Discrete): Joueur dont c'est le tour (0 ou 1). Non utilisé par l'agent
                                 si l'agent voit toujours l'état de son propre point de vue.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.verbose = render_mode == 'human'
        self.render_mode = render_mode

        self._num_tile_actions = NUM_TILE_TYPES
        self._play_action_offset = PLAY_ACTION_OFFSET
        self._action_draw = ACTION_DRAW
        self._action_pass = ACTION_PASS
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        obs_components = {
            'my_hand': spaces.MultiBinary(self._num_tile_actions),

            'open_ends': spaces.Box(low=-1, high=MAX_DOMINO_VALUE, shape=(2,), dtype=np.float32),

            'board_value_counts': spaces.Box(low=0, high=(MAX_DOMINO_VALUE + 2), shape=(MAX_DOMINO_VALUE + 1,), dtype=np.float32),

            'opponent_hand_size': spaces.Discrete(TOTAL_TILES + 1),
            'draw_pile_size': spaces.Discrete(TOTAL_TILES - 2 * HAND_SIZE + 1),

          
            'current_player_id': spaces.Discrete(2),
        }
        self.observation_space = spaces.Dict(obs_components)

        self.player_hands: List[List[DominoTile]] = [[], []]
        self.draw_pile: List[DominoTile] = []
        self.board: List[DominoTile] = []
        self.open_ends: List[int] = []
        self.current_player: int = 0
        self.scores: List[int] = [0, 0]
        self.history: List[Tuple[int, Any, Optional[int]]] = []
        self.done: bool = False
        self.winner: int = -1
        self.consecutive_passes: int = 0
        self.turn_count: int = 0
        self.last_action_description: str = "N/A"

    def _get_obs(self) -> Dict[str, Any]:
        """Construit et retourne l'observation actuelle pour l'agent."""

        my_hand_encoding = np.zeros(self._num_tile_actions, dtype=np.int8)
        for tile in self.player_hands[self.current_player]:
            idx = DOMINO_TO_INDEX.get(tile)
            if idx is not None:
                my_hand_encoding[idx] = 1

        ends = sorted(self.open_ends)
        if len(ends) == 0:
            open_ends_encoding = np.array([-1.0, -1.0], dtype=np.float32)
        elif len(ends) == 1:
            open_ends_encoding = np.array([ends[0], -1.0], dtype=np.float32)
        else: 
            open_ends_encoding = np.array(ends, dtype=np.float32)

        board_value_counts = np.zeros(MAX_DOMINO_VALUE + 1, dtype=np.float32)
        for tile in self.board:
            s1, s2 = tile.get_sides()
            board_value_counts[s1] += 1.0
            if not tile.is_double():
                board_value_counts[s2] += 1.0
        # max_count_per_value = (MAX_DOMINO_VALUE + 2)
        # board_value_counts /= max_count_per_value

        # 4. Tailles et joueur
        opponent_player = 1 - self.current_player
        opponent_hand_size = np.array(len(self.player_hands[opponent_player]), dtype=np.int32)
        draw_pile_size = np.array(len(self.draw_pile), dtype=np.int32)
        current_player_id = np.array(self.current_player, dtype=np.int32)

        return {
            'my_hand': my_hand_encoding,
            'open_ends': open_ends_encoding,
            'board_value_counts': board_value_counts,
            'opponent_hand_size': opponent_hand_size,
            'draw_pile_size': draw_pile_size,
            'current_player_id': current_player_id,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Retourne des informations de débogage/diagnostique."""
        return {
            "player_0_hand": self.player_hands[0][:],
            "player_1_hand": self.player_hands[1][:],
            "board": self.board[:],
            "open_ends": self.open_ends[:],
            "draw_pile_size": len(self.draw_pile),
            "consecutive_passes": self.consecutive_passes,
            "scores": self.scores[:],
            "last_action_desc": self.last_action_description,
            "winner": self.winner,
            "turn_count": self.turn_count
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Réinitialise le jeu."""
        super().reset(seed=seed)

        logger.info("Réinitialisation de la partie de domino")
        shuffled_tiles = ALL_DOMINOS[:]
        self.np_random.shuffle(shuffled_tiles) 

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
        self.turn_count = 0
        self.last_action_description = "Game Start"

        if self.verbose:
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        # logger.debug(f"Reset complete. Player {self.current_player} starts. Obs: {observation}, Info: {info}")
        return observation, info

    def _determine_first_player(self) -> int:
        """Détermine qui commence (double le plus élevé, sinon domino le plus fort)."""
        highest_double = -1
        first_player = -1
        start_tile = None 

        for player_idx, hand in enumerate(self.player_hands):
            for tile in hand:
                if tile.is_double():
                    if tile.side1 > highest_double:
                        highest_double = tile.side1
                        first_player = player_idx
                        start_tile = tile

        if first_player == -1: 
            highest_value = -1
            highest_side = -1
            for player_idx, hand in enumerate(self.player_hands):
                for tile in hand:
                    tile_val = tile.get_value()
                    tile_max_side = max(tile.get_sides())
                    if tile_val > highest_value or (tile_val == highest_value and tile_max_side > highest_side):
                        highest_value = tile_val
                        highest_side = tile_max_side
                        first_player = player_idx
                        start_tile = tile

        if first_player == -1:
            first_player = self.np_random.integers(0, 2) 

        logger.info(f"Le joueur {first_player} commence la partie (avec {start_tile} comme critère).")
        return first_player

    def _decode_action_code(self, action_code: int) -> Dict[str, Any]:
        """ Traduit un code d'action (0-57) en détails d'action logique. """
        if not (0 <= action_code < self.action_space.n):
             raise ValueError(f"Code d'action invalide {action_code}, hors de [0, {self.action_space.n-1}]")

        if action_code == self._action_draw:
            return {"type": "draw"}
        if action_code == self._action_pass:
            return {"type": "pass"}

        target_end_index = -1
        tile_index = -1      
        chosen_end_value = -1

        if ACTION_PLAY_END_0_START <= action_code < ACTION_PLAY_END_0_END:
            target_end_index = 0
            tile_index = action_code
        elif ACTION_PLAY_END_1_START <= action_code < ACTION_PLAY_END_1_END:
            target_end_index = 1
            tile_index = action_code - self._play_action_offset
        else:
            raise ValueError(f"Code d'action {action_code} invalide pour une action 'play'.")

        tile_to_play = INDEX_TO_DOMINO.get(tile_index)
        if tile_to_play is None:
            raise ValueError(f"Index de tuile {tile_index} invalide.")

       
        current_hand = self.player_hands[self.current_player]
        try:
            hand_idx = current_hand.index(tile_to_play)
        except ValueError:
             raise RuntimeError(f"Action {action_code} ({tile_to_play}) décodée mais tuile non en main {current_hand}!")

        s1, s2 = tile_to_play.get_sides()
        flip = False

        if not self.board:
             chosen_end_value = -1 
             flip = False
        else:
            if len(self.open_ends) == 0:
                 raise RuntimeError("Tentative de jouer alors que le plateau a des tuiles mais pas d'extrémités ouvertes!")
            if target_end_index >= len(self.open_ends):
                 if target_end_index == 1 and len(self.open_ends) == 1:
                      raise ValueError(f"Action {action_code} vise l'extrémité 1, mais seule l'extrémité 0 existe.")
                 raise ValueError(f"Index d'extrémité cible {target_end_index} invalide pour open_ends {self.open_ends}")

            chosen_end_value = self.open_ends[target_end_index]

            if s1 == chosen_end_value:
                flip = False
            elif s2 == chosen_end_value:
                flip = True
            else:
                 raise RuntimeError(f"Action {action_code} ({tile_to_play}) décodée pour l'extrémité {target_end_index} ({chosen_end_value}), mais aucune face ne correspond!")

        return {"type": "play", "tile": tile_to_play, "hand_idx": hand_idx, "end_value": chosen_end_value, "flip": flip}

    def step(self, action_code: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """ Exécute une étape du jeu en utilisant un code d'action (0-57). """
        if self.done:
            logger.warning("step() appelé sur un environnement terminé.")
            info = self._get_info()
            info['action_details'] = {'type': 'noop_done'}
            return self._get_obs(), 0.0, True, False, info

        action_details: Dict[str, Any] = {}
        try:
            legal_mask = self.get_legal_action_mask()
            if not legal_mask[action_code]:
                 raise ValueError(f"Action {action_code} choisie mais masque légal est False.")

            action_details = self._decode_action_code(action_code)
        except ValueError as e:
             logger.warning(f"Action illégale tentée par joueur {self.current_player}: Code {action_code}. Masque légal: {self.get_legal_action_mask()}. Erreur: {e}")
             current_obs = self._get_obs()
             info = self._get_info()
             info['action_details'] = {'type': 'invalid_action', 'code': action_code, 'error': str(e)}
             return current_obs, 0.0, False, False, info 
        except Exception as e:
             logger.exception(f"Erreur inattendue lors du traitement de action_code {action_code}: {e}")
             self.done = True
             info = self._get_info()
             info['action_details'] = {'type': 'critical_error', 'code': action_code}
             return self._get_obs(), -10.0, True, False, info

        action_type = action_details.get("type")
        player = self.current_player
        reward = 0.0
        terminated = False
        truncated = False

        self.last_action_description = f"Player {player}: Attempt {action_type}"
        if action_type == "play": self.last_action_description += f" tile {action_details.get('tile')} matching end {action_details.get('end_value')}"
        elif action_type == "draw": self.last_action_description += " draw"
        elif action_type == "pass": self.last_action_description += " pass"

       
        if action_type == "play":
            tile_to_play = action_details["tile"]
            hand_idx = action_details["hand_idx"]
            end_to_match = action_details["end_value"] 
            flip = action_details["flip"]
            hand = self.player_hands[player]

            
            actual_tile_played = hand.pop(hand_idx)
            assert actual_tile_played == tile_to_play 

            self.last_action_description = f"Player {player}: Played {tile_to_play}"
            if end_to_match != -1: self.last_action_description += f" matching {end_to_match}"

            s1, s2 = tile_to_play.get_sides()
            if not self.board: 
                self.board.append(tile_to_play)
                self.open_ends = [s1, s2]
            else:
                connected_side = s2 if flip else s1
                new_open_side = s1 if flip else s2

                if connected_side != end_to_match:
                     logger.error(f"Incohérence interne: Joué {tile_to_play} (flip={flip}) pour matcher {end_to_match}, mais côté connecté calculé est {connected_side}")
                     hand.insert(hand_idx, tile_to_play)
                     self.done = True
                     info = self._get_info()
                     info['action_details'] = {'type': 'critical_flip_error', 'code': action_code}
                     return self._get_obs(), -10.0, True, False, info

                try:
                    idx_to_replace = self.open_ends.index(end_to_match)
                    self.open_ends[idx_to_replace] = new_open_side
                except ValueError:
                     if len(self.open_ends) == 2 and self.open_ends[0] == self.open_ends[1] and self.open_ends[0] == end_to_match:
                          self.open_ends[0] = new_open_side
                     else:
                          logger.error(f"Erreur critique: Extrémité {end_to_match} non trouvée dans {self.open_ends} pour jouer {tile_to_play}")
                          hand.insert(hand_idx, tile_to_play) 
                          self.done = True
                          info = self._get_info()
                          info['action_details'] = {'type': 'critical_board_error', 'code': action_code}
                          return self._get_obs(), -10.0, True, False, info

            self.board.append(tile_to_play)
            self.history.append((player, tile_to_play, end_to_match))
            self.consecutive_passes = 0

            if not hand:
                terminated = True
                self.winner = player
                reward = self._calculate_reward(player)
                self.last_action_description += " -> WIN!"
                logger.info(f"Joueur {player} a gagné!")
            else:
                self.current_player = 1 - player
                self.turn_count += 1

        elif action_type == "draw":
            drawn_tile = self.draw_pile.pop(0)
            self.player_hands[player].append(drawn_tile)
            self.history.append((player, "draw", None))
            self.consecutive_passes = 0
            self.last_action_description = f"Player {player}: Drew {drawn_tile}. Hand={len(self.player_hands[player])}"
            logger.info(f"Joueur {player} pioche {drawn_tile}")

        elif action_type == "pass":
             self.history.append((player, "pass", None))
             self.consecutive_passes += 1
             self.last_action_description = f"Player {player}: Passed (legal)"
             logger.info(f"Joueur {player} passe son tour (légal)")

             if self.consecutive_passes >= 2:
                  terminated = True
                  reward = self._calculate_reward_blocked(player)
                  self.last_action_description += " -> BLOCKED GAME!"
                  logger.info(f"Jeu bloqué! Scores finaux mains: P0={self._get_hand_value(0)}, P1={self._get_hand_value(1)}. Gagnant: {self.winner}")
             else:
                  self.current_player = 1 - player
                  self.turn_count += 1

        else:
             logger.error(f"Type d'action inconnu '{action_type}' dans step() après décodage.")
             self.done = True
             info = self._get_info()
             info['action_details'] = {'type': 'unknown_action_type', 'received_type': action_type}
             return self._get_obs(), -10.0, True, False, info

        self.done = terminated or truncated

        if self.verbose and not self.done:
             self.render()

        observation = self._get_obs()
        info = self._get_info()
        info['action_details'] = action_details

        final_reward_for_acting_player = reward

        return observation, final_reward_for_acting_player, terminated, truncated, info

    def _get_hand_value(self, player_idx: int) -> int:
        """Calcule la somme des points dans la main d'un joueur."""
        return sum(tile.get_value() for tile in self.player_hands[player_idx])

    def _calculate_reward(self, winner: int) -> float:
        """Calcule la récompense standard (+1/-1) quand un joueur gagne."""
        if winner == -1: return 0.0 # Nul
        acting_player = self.current_player
        if winner == acting_player: return 1.0
        else: return -1.0 

    def _calculate_reward_blocked(self, acting_player: int) -> float:
        """Détermine le gagnant par blocage et la récompense pour le joueur qui vient de passer."""
        score0 = self._get_hand_value(0)
        score1 = self._get_hand_value(1)

        if score0 < score1: self.winner = 0
        elif score1 < score0: self.winner = 1
        else: self.winner = -1 

        if self.winner == acting_player: return 1.0
        elif self.winner == -1: return 0.0
        else: return -1.0

    def render(self):
        """Affiche l'état du jeu en mode console."""
        if self.render_mode != "human": return

        print("\n" + "="*60)
        print(f"Dernière Action: {self.last_action_description}")
        print(f"Tour du Joueur: {self.current_player} | Tour N°: {self.turn_count}")
        board_str = " ".join(map(str, self.board))
        print(f"Plateau ({len(self.board)} tuiles): {board_str}")
        print(f"Extrémités Ouvertes: {self.open_ends}")
        print(f"Main Joueur 0 ({len(self.player_hands[0])} tuiles): {sorted(self.player_hands[0])}")
        print(f"Main Joueur 1 ({len(self.player_hands[1])} tuiles): {sorted(self.player_hands[1])}")
        print(f"Pioche: {len(self.draw_pile)} tuiles")
        print(f"Passes Consécutives: {self.consecutive_passes}")
        print("-" * 20)
        # legal_map = self._get_legal_actions_map()
        # print("Actions Légales Codes:", list(legal_map.keys()))
        # print("Actions Légales Détails:", legal_map)

        if self.done:
             print("*" * 20 + " FIN DE PARTIE " + "*" * 20)
             winner_text = f"Joueur {self.winner}" if self.winner != -1 else "Match Nul"
             print(f"Gagnant: {winner_text}")
             print(f"Score Main Joueur 0: {self._get_hand_value(0)}")
             print(f"Score Main Joueur 1: {self._get_hand_value(1)}")
        print("="*60 + "\n")

    def close(self):
        """Nettoyage à la fermeture de l'environnement."""
        pass 
    def get_legal_action_mask(self) -> np.ndarray:
        """
        Retourne un masque booléen/binaire pour les actions légales (taille 58).
        Index i est True si l'action i est légale.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        hand = self.player_hands[self.current_player]
        can_play_any_tile = False

        if not self.board: 
            for tile in hand:
                tile_idx = DOMINO_TO_INDEX.get(tile)
                if tile_idx is not None:
                    mask[ACTION_PLAY_END_0_START + tile_idx] = True
                    can_play_any_tile = True
        else:
            if not self.open_ends:
                 logger.error("Erreur: Plateau non vide mais pas d'extrémités ouvertes!")
                 pass
            else:
                 end0_value = self.open_ends[0]
                 end1_value = self.open_ends[1] if len(self.open_ends) > 1 else -1 

                 # Si les deux extrémités sont identiques (ex: après [6|6])
                 ends_are_same = len(self.open_ends) == 2 and end0_value == end1_value

                 for tile in hand:
                      tile_idx = DOMINO_TO_INDEX.get(tile)
                      if tile_idx is None: continue 

                      s1, s2 = tile.get_sides()
                      playable_on_end0 = (s1 == end0_value or s2 == end0_value)
                      playable_on_end1 = (len(self.open_ends) > 1) and (s1 == end1_value or s2 == end1_value)

                      if playable_on_end0:
                           mask[ACTION_PLAY_END_0_START + tile_idx] = True
                           can_play_any_tile = True
                           if ends_are_same:
                                playable_on_end1 = False # Empêcher de marquer l'action pour end1

                      if playable_on_end1:
                           mask[ACTION_PLAY_END_1_START + tile_idx] = True
                           can_play_any_tile = True

        if not can_play_any_tile:
            if self.draw_pile:
                mask[self._action_draw] = True
            else:
                mask[self._action_pass] = True

        if not np.any(mask):
             logger.error(f"Aucune action légale trouvée! État: Joueur {self.current_player}, Main {hand}, Plateau {self.board}, Extrémités {self.open_ends}, Pioche {len(self.draw_pile)}")
             mask[self._action_pass] = True

        return mask