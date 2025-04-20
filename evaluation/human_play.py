import sys
from environment.domino_env import DominoEnv
from agents.ppo_agent import PPOTrainer

class HumanInterface:
    def __init__(self, model_path="models/best_model.pt"):
        self.env = DominoEnv()
        self.agent = PPOTrainer(self.env)
        self.agent.load_checkpoint(model_path)
        
    def _display_hand(self, hand):
        """Affiche la main de façon lisible"""
        print("\nVotre main:")
        for i, domino in enumerate(hand):
            print(f"{i+1}: [{domino[0]}|{domino[1]}]")

    def _display_board(self):
        """Affiche le plateau de jeu"""
        print("\n=== Plateau ===")
        chain = "-".join(f"[{d[0]}|{d[1]}]" for d in self.env.board)
        print(chain + "\n")

    def _human_turn(self):
        """Gère le tour du joueur humain"""
        valid_actions = self.env._get_valid_actions_mask()
        hand = self.env.players[self.env.current_player]
        
        self._display_hand(hand)
        self._display_board()
        
        while True:
            try:
                choice = int(input("Choisissez un domino (numéro) : ")) - 1
                if 0 <= choice < len(hand) and valid_actions[self.env.domino_to_id(hand[choice])]:
                    return self.env.domino_to_id(hand[choice])
                print("Coup invalide! Réessayez.")
            except ValueError:
                print("Entrez un nombre valide!")

    def play(self):
        """Lance une partie complète"""
        obs = self.env.reset()
        self._display_board()
        
        while not self.env.done:
            if self.env.current_player == 0:  # Humain
                action = self._human_turn()
            else:  # IA
                action = self.agent.get_action(obs, training=False)
                print(f"L'IA joue: {self.env.id_to_domino(action)}")
                
            obs, _, done, _ = self.env.step(action)
            self._display_board()

        if self.env.winner == 0:
            print("Félicitations! Vous avez gagné!")
        else:
            print("L'IA a gagné. Essayez encore!")