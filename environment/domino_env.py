import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Dict, Discrete, Box

class DominoEnv(Env):
    def __init__(self):
        # Configuration de base
        self.num_players = 2
        self.domino_set = [(i, j) for i in range(7) for j in range(i, 7)]
        
        # Espaces d'observation/action
        self.action_space = Discrete(28)
        self.observation_space = Dict({
            "hand": Box(0, 1, (28,)),         # Main du joueur (one-hot)
            "board": Box(0, 6, (20, 2)),       # Chaîne actuelle (max 20 dominos)
            "valid_actions": Box(0, 1, (28,)), # Masque d'actions valides
            "hidden_state": Box(0, 1, (15,))   # État caché estimé (15 features)
        })
        
        self.reset()

    def reset(self):
        # Mélange et distribution des dominos
        np.random.shuffle(self.domino_set)
        self.players = {
            0: self.domino_set[:7],
            1: self.domino_set[7:14]
        }
        
        # Trouver le joueur qui commence
        self.current_player = self._find_starting_player()
        self.board = []
        self.done = False
        
        # Poser le double initial
        starting_domino = max([d for d in self.players[self.current_player] if d[0] == d[1]], default=None)
        if starting_domino:
            self._play_domino(starting_domino)
        
        return self._get_obs()

    def _get_obs(self):
        """Construction de l'observation avec partial observability"""
        return {
            "hand": self._encode_hand(),
            "board": self._encode_board(),
            "valid_actions": self._get_valid_actions_mask(),
            "hidden_state": self._estimate_hidden_state()
        }

    def _encode_hand(self):
        # Encodage one-hot de la main
        hand = np.zeros(28)
        for domino in self.players[self.current_player]:
            hand[self.domino_to_id(domino)] = 1.0
        return hand

    def _encode_board(self):
        # Encodage de la chaîne de dominos
        board = np.zeros((20, 2))
        for i, d in enumerate(self.board):
            board[i] = d
        return board

    def _get_valid_actions_mask(self):
        # Génération du masque d'actions valides
        mask = np.zeros(28)
        if not self.board:  # Premier coup
            for d in self.players[self.current_player]:
                if d[0] == d[1]:  # Seuls les doubles sont valides initialement
                    mask[self.domino_to_id(d)] = 1.0
        else:
            left, right = self.board[0][0], self.board[-1][1]
            for d in self.players[self.current_player]:
                if d[0] in (left, right) or d[1] in (left, right):
                    mask[self.domino_to_id(d)] = 1.0
        return mask

    def _estimate_hidden_state(self):
        """Estimation de l'état caché (opponent's hand et dominos restants)"""
        # Implémentation basique (à améliorer avec belief network)
        played = set(self.domino_to_id(d) for d in self.board)
        possible = np.ones(28)
        possible[list(played)] = 0
        return possible

    def step(self, action):
        # Exécution d'une action
        reward = 0
        info = {}
        
        # Validation de l'action
        if self._get_valid_actions_mask()[action] == 0:
            reward -= 5  # Pénalité pour coup invalide
            return self._get_obs(), reward, self.done, info
        
        # Jouer le domino
        domino = self.id_to_domino(action)
        self._play_domino(domino)
        
        # Calcul de la récompense
        reward += self._calculate_reward(domino)
        
        # Vérification fin de partie
        if len(self.players[self.current_player]) == 0:
            self.done = True
            reward += 100  # Bonus pour victoire
            
        return self._get_obs(), reward, self.done, info

    def _calculate_reward(self, domino):
        """Reward shaping basé sur Ng et al. 1999"""
        # 1. Récompense pour réduire le nombre de points
        score = sum(domino)
        
        # 2. Pénalité pour garder des dominos forts
        hand_values = [sum(d) for d in self.players[self.current_player]]
        penalty = max(hand_values) * 0.1
        
        # 3. Bonus pour bloquer l'adversaire
        # (À implémenter avec opponent model)
        
        return (score - penalty) * 0.1
   
    def domino_to_id(self, domino):
        """Convertit un domino (a,b) en ID unique entre 0 et 27"""
        a, b = sorted(domino)  # Garantir l'ordre croissant
        return (a * (a + 1)) // 2 + b

    def id_to_domino(self, domino_id):
        """Convertit un ID en domino (a,b)"""
        a = 0
        while (a + 1) * (a + 2) // 2 <= domino_id:
            a += 1
        b = domino_id - (a * (a + 1)) // 2
        return (a, b) if a <= b else (b, a)
    
    def _find_starting_player(self):
        """Trouve le joueur avec le double le plus élevé"""
        max_double = -1
        starter = 0
        
        for player in [0, 1]:
            for domino in self.players[player]:
                if domino[0] == domino[1]:  # C'est un double
                    if domino[0] > max_double:
                        max_double = domino[0]
                        starter = player
                        
        return starter if max_double != -1 else 0
    
    def _play_domino(self, domino):
        """Joue un domino sur le plateau et met à jour l'état du jeu"""
        # Retirer le domino de la main du joueur
        self.players[self.current_player].remove(domino)
        
        # Ajouter au plateau selon les règles de positionnement
        if not self.board:
            self.board.append(domino)
        else:
            left, right = self.board[0][0], self.board[-1][1]
            a, b = domino
            
            # Vérifier les deux orientations possibles
            if a == left:
                self.board.insert(0, (b, a))  # Inverser pour orientation gauche
            elif b == left:
                self.board.insert(0, domino)
            elif a == right:
                self.board.append(domino)
            elif b == right:
                self.board.append((b, a))  # Inverser pour orientation droite
            else:
                raise ValueError("Domino invalide - devrait être validé avant")
    
    def _is_blocked(self):
        """Vérifie si le jeu est bloqué pour les deux joueurs"""
        current_valid = np.sum(self._get_valid_actions_mask())
        
        # Vérifier pour l'autre joueur
        other_player = 1 - self.current_player
        original_player = self.current_player
        self.current_player = other_player
        other_valid = np.sum(self._get_valid_actions_mask())
        self.current_player = original_player
        
        return current_valid == 0 and other_valid == 0
    
    def _calculate_final_scores(self):
        """Calcule les scores finaux en cas de blocage"""
        if not self._is_blocked():
            return {}
            
        scores = {
            0: sum(sum(d) for d in self.players[0]),
            1: sum(sum(d) for d in self.players[1])
        }
        
        # Déterminer le gagnant
        min_score = min(scores.values())
        self.winner = [p for p, s in scores.items() if s == min_score][0]
        
        return scores
    
    def _update_hidden_state(self):
        """Met à jour la représentation de l'état caché"""
        played_dominoes = set(self.domino_to_id(d) for d in self.board)
        all_dominoes = set(range(28))
        
        # Dominos non joués = main adverse + dominos non distribués
        remaining = list(all_dominoes - played_dominoes)
        
        # Créer une distribution de probabilité (simplifiée)
        self.hidden_state = np.zeros(28)
        for d in remaining:
            self.hidden_state[d] = 1.0 / len(remaining)
    
    def _calculate_reward(self, domino):
        """Reward shaping avancé selon Ng et al. 1999"""
        base_reward = sum(domino) * 0.1  # Récompense immédiate
        
        # Pénalité pour les dominos forts restants
        hand_strength = sum(sum(d) for d in self.players[self.current_player])
        penalty = hand_strength * 0.05
        
        # Bonus stratégique pour les options futures
        future_options = len(self._get_valid_actions_mask().nonzero()[0])
        flexibility_bonus = future_options * 0.2
        
        # Bonus de victoire (ajusté dans step())
        victory_bonus = 100 if len(self.players[self.current_player]) == 0 else 0
        
        return base_reward - penalty + flexibility_bonus + victory_bonus
    
    def render(self, mode='human'):
        """Affichage ASCII du plateau de jeu"""
        print(f"\nJoueur actuel: {self.current_player}")
        print("Main du joueur:")
        for d in sorted(self.players[self.current_player]):
            print(f"[{d[0]}|{d[1]}]", end=" ")
        
        print("\n\nPlateau:")
        chain = "-".join(f"[{d[0]}|{d[1]}]" for d in self.board)
        print(chain + "\n")
    
    