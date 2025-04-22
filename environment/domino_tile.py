import functools

@functools.total_ordering 
class DominoTile:
    """Représente un domino avec deux côtés."""

    def __init__(self, side1: int, side2: int):
        self.side1 = min(side1, side2)
        self.side2 = max(side1, side2)

    def __repr__(self):
        return f"[{self.side1}|{self.side2}]"

    def __eq__(self, other):
        if not isinstance(other, DominoTile):
            return NotImplemented 
        return (self.side1, self.side2) == (other.side1, other.side2)

    def __lt__(self, other):
        if not isinstance(other, DominoTile):
            return NotImplemented
        return (self.side1, self.side2) < (other.side1, other.side2)

    def __hash__(self):
        return hash((self.side1, self.side2))

    def matches(self, value: int) -> bool:
        """Vérifie si le domino peut être joué contre une valeur."""
        return self.side1 == value or self.side2 == value

    def get_other_side(self, value: int) -> int:
        """Retourne la valeur de l'autre côté du domino."""
        if self.side1 == value:
            return self.side2
        elif self.side2 == value:
            return self.side1
        else:
            raise ValueError(f"La valeur {value} n'est pas présente dans ce domino {self}")

    def is_double(self) -> bool:
        """Vérifie si le domino est un double."""
        return self.side1 == self.side2

    def get_value(self) -> int:
        """Retourne la valeur totale (somme des points) du domino."""
        return self.side1 + self.side2

    def get_sides(self) -> tuple[int, int]:
        """Retourne les deux côtés du domino."""
        return self.side1, self.side2