import functools
from typing import Tuple
@functools.total_ordering
class DominoTile:
    """
    Représente un domino unique avec deux côtés numériques.

    La classe assure une représentation canonique où side1 <= side2,
    ce qui facilite les comparaisons et l'utilisation dans des collections.

    Attributs:
        side1 (int): La valeur du premier côté (garanti <= side2).
        side2 (int): La valeur du second côté (garanti >= side1).
    """

    def __init__(self, side1: int, side2: int):
        """
        Initialise un DominoTile. Assure la représentation canonique side1 <= side2.
        """
        self.side1 = min(side1, side2)
        self.side2 = max(side1, side2)

    def __repr__(self) -> str:
        """Retourne une représentation textuelle du domino (ex: "[3|6]")."""
        return f"[{self.side1}|{self.side2}]"

    def __eq__(self, other) -> bool:
        """Vérifie l'égalité basée sur les côtés canoniques."""
        if not isinstance(other, DominoTile):
            return NotImplemented
        return (self.side1, self.side2) == (other.side1, other.side2)

    def __lt__(self, other) -> bool:
        """Définit l'ordre lexicographique pour le tri."""
        if not isinstance(other, DominoTile):
            return NotImplemented
        return (self.side1, self.side2) < (other.side1, other.side2)

    def __hash__(self) -> int:
        """Calcule le hash basé sur les côtés canoniques."""
        return hash((self.side1, self.side2))

    def matches(self, value: int) -> bool:
        """Vérifie si la 'value' correspond à l'un des côtés."""
        return self.side1 == value or self.side2 == value

    def get_other_side(self, value: int) -> int:
        """
        Retourne la valeur de l'autre côté si 'value' correspond à un côté.
        Lève une ValueError si 'value' n'est pas sur le domino.
        """
        if self.side1 == value:
            return self.side2
        elif self.side2 == value:
            return self.side1
        else:
            raise ValueError(f"La valeur {value} n'est pas présente sur le domino {self}")

    def is_double(self) -> bool:
        """Retourne True si c'est un double."""
        return self.side1 == self.side2

    def get_value(self) -> int:
        """Retourne la somme des points des deux côtés."""
        return self.side1 + self.side2

    def get_sides(self) -> Tuple[int, int]:
        """Retourne un tuple (side1, side2)."""
        return (self.side1, self.side2)