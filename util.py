import numpy as np

def format_time(seconds: float, acceleration: float) -> str:
    if seconds < 0:
        return "сек: 0"

    units = [
        (31536000, "лет"),
        (86400, "дней"),
        (3600, ""),
        (60, ""),
        (1, "")
    ]

    if acceleration > 1000000:
        units = units[:2]
    elif acceleration > 10000:
        units = units[:3]
    elif acceleration > 100:
        units = units[:4]

    result = []
    remaining = seconds

    for value, prefix in units:
        if remaining >= value:
            count = int(remaining // value)
            remaining %= value
            result.append(f"{prefix}: {count:02d}")

    return " ".join(result) if result else "сек: 0"

class NVector:
    def __init__(self, x=0, y=0):
        if isinstance(x, NVector):
            # Если x это NVector, копируем его array
            self.array = x.array.copy()
        elif hasattr(x, '__len__') and len(x) == 2:
            # Если x это последовательность длины 2
            self.array = np.array([x[0], x[1]], dtype=float)
        else:
            # Если x и y это отдельные координаты
            self.array = np.array([float(x), float(y)], dtype=float)

    @property
    def x(self):
        return self.array[0]

    @x.setter
    def x(self, value):
        self.array[0] = float(value)

    @property
    def y(self):
        return self.array[1]

    @y.setter
    def y(self, value):
        self.array[1] = float(value)

    def __add__(self, other):
        if isinstance(other, NVector):
            return NVector(*(self.array + other.array))
        return NVector(*(self.array + other))

    def __sub__(self, other):
        if isinstance(other, NVector):
            return NVector(*(self.array - other.array))
        return NVector(*(self.array - other))

    def __mul__(self, scalar):
        return NVector(*(self.array * scalar))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return NVector(*(self.array / scalar))

    def __neg__(self):
        return NVector(*(-self.array))

    def __eq__(self, other):
        if not isinstance(other, NVector):
            return False
        return np.allclose(self.array, other.array)

    def __str__(self):
        return f"NVector({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()

    def length(self):
        return np.sqrt(np.sum(self.array * self.array))

    def length_squared(self):
        return np.sum(self.array * self.array)

    def normalize(self):
        length = self.length()
        if length != 0:
            return NVector(*(self.array / length))
        return NVector(*self.array)

    def dot(self, other):
        if isinstance(other, NVector):
            return np.dot(self.array, other.array)
        return np.dot(self.array, other)

    def rotate_rad(self, angle):
        """Поворачивает 2D вектор на заданный угол в радианах"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x = self.array[0] * cos_a - self.array[1] * sin_a
        y = self.array[0] * sin_a + self.array[1] * cos_a
        return NVector(x, y)

    @classmethod
    def from_array(cls, array):
        if len(array) != 2:
            raise ValueError("Array must have length 2")
        return cls(array[0], array[1])

