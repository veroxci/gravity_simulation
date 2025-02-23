import pygame
import pygame.gfxdraw
import math
import numpy as np
from util import NVector
from collections import deque
import random
import colorsys
from typing import Deque, Tuple
from constants import *

def generate_distinct_color() -> Tuple[int, int, int]:
    # Генерируем случайный яркий цвет
    hue = random.random()
    saturation = 0.8 + random.random() * 0.2 # 0.2 - 0.8
    value = 0.8 + random.random() * 0.2
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    # В функциях colorsys значения каждого канала от 0 до 1, приведем их к стандартным
    return int(r * 255), int(g * 255), int(b * 255)

def rough_density_by_mass(mass):
    """
    Возвращаем примерную плотность небесного тела (в кг/м³) по его массе (в кг)
    Аппроксимация для разных типов тел:
    1. Малые тела (плотность около кремния ~2330 кг/м³)
    2. Планеты земной группы (~5500 кг/м³ как Земля)
    3. Газовые гиганты (~1000 кг/м³)
    4. Звёзды (~1400 кг/м³ как Солнце)
    """

    if mass < 1e15:  # меньше триллиона тонн
        return 2500  # примерная плотность обычных твёрдых тел

    log_mass = np.log10(mass)
    base_density = 5500

    # Формула с плавным уменьшением плотности к переходам к большим массам
    if log_mass <= 24.5:  # До массы Земли
        relative_density = 1.0
    else:
        # Постепенное убывание от плотности планет земной группы к плотности звёзд
        decay_rate = 0.15
        target_density = 1400  # примерная плотность Солнца
        relative_density = 1.0 + (target_density / base_density - 1.0) * (1 - np.exp(-decay_rate * (log_mass - 24.5)))

    return max(base_density * relative_density, 1000)  # Минимальная плотность 1000 кг/м³

class Body:
    def __init__(self, pos: NVector, mass: float, velocity: NVector): # Инициализация нового тела
        self.radius = MIN_BODY_RADIUS # Сначала создаём тело с минимальным радиусом
        self.pos = pos
        self.mass = mass
        self.velocity = velocity
        self.acceleration = NVector(0, 0) # тело только появилось, и ускорения у него еще нет
        self.trajectory: Deque[np.array] = deque(maxlen=TRAJECTORY_LENGTH)
        self.color = generate_distinct_color()
        self.selected = False
        self.is_fragment = False  # флаг осколка
        self.fragment_time = 0  # время создания осколка
        self.update_radius() # теперь обновим радиус, чтобы не было разницы с изменением через интерфейс

    def update_radius(self):
        # Выставляем радиус тела в зависимости от массы тела
        radius = (self.mass / rough_density_by_mass(self.mass)*3/math.pi/4) ** (1/3)
        self.radius = max(MIN_BODY_RADIUS, radius)
    
    def clear_trajectory(self):
        # Очищаем запомненную траекторию
        self.trajectory.clear()
    
    def update_trajectory(self, scale):
        # Добавляем текущую позицию к предыдущим точкам траектории
        p = self.pos.array
        if (len(self.trajectory) == 0
                or abs(self.trajectory[-1][0] - p[0]) * scale  > 2
                or abs(self.trajectory[-1][1] - p[1]) * scale  > 2):
            self.trajectory.append(np.copy(self.pos.array))

    def get_state_vector(self) -> np.ndarray:
        # Возвращаем вектор состояния для численного интегрирования
        return np.array([self.pos.array[0], self.pos.array[1], self.velocity.array[0], self.velocity.array[1]])
    
    def set_state_vector(self, state: np.ndarray):
        # Устанавливаем состояние из вектора состояния
        self.pos.array[0] = state[0]
        self.pos.array[1] = state[1]
        self.velocity.array[0] = state[2]
        self.velocity.array[1] = state[3]

    def draw_thick_aaline(self, screen, color, start_pos, end_pos, thickness=3):
        # Pygame не умеет рисовать сглаженные линии толще одного пикселя, поэтому, чтобы получить красивую и хорошо видимую рисуем рядом несколько однопиксельных
        delta = end_pos - start_pos

        if delta.length_squared() == 0:
            return

        # Перпендикулярный вектор единичной длины
        nperp = delta.rotate_rad(math.pi/2).normalize()

        # Чтобы получить красивую толстую линию рисуем нужное кол-во параллельных линий (по величине thickness)
        for i in range(thickness):
            offset = (i - thickness / 2 + 0.5)
            p1 = (start_pos.x + nperp.x * offset, start_pos.y + nperp.y * offset)
            p2 = (end_pos.x + nperp.x * offset, end_pos.y + nperp.y* offset)
            pygame.draw.aaline(screen, color, p1, p2)

    def draw_velocity_vector(self, screen, screen_pos: NVector):
        # Тут отрисовка вектора скорости
        if self.velocity.length() > 0:
            # Определяем длину вектора скорости
            vel_length = max(20, int(math.log10(self.velocity.length() + 1) * 10))
            vel_dir = self.velocity.normalize().array
            end_pos = screen_pos + vel_dir * vel_length

            # Основная линия вектора
            self.draw_thick_aaline(screen, (255, 0, 0), screen_pos, end_pos, 3)

            # Стрелка
            arrow_size = 8  # Немного увеличим размер для толстой линии
            angle = math.pi / 9
            dx = end_pos.x - screen_pos.x
            dy = end_pos.y - screen_pos.y
            main_angle = math.atan2(dy, dx)

            arrow1_x = end_pos.x - arrow_size * math.cos(main_angle + angle)
            arrow1_y = end_pos.y - arrow_size * math.sin(main_angle + angle)
            arrow2_x = end_pos.x - arrow_size * math.cos(main_angle - angle)
            arrow2_y = end_pos.y - arrow_size * math.sin(main_angle - angle)

            arrow1 = NVector(arrow1_x, arrow1_y)
            arrow2 = NVector(arrow2_x, arrow2_y)

            self.draw_thick_aaline(screen, (255, 0, 0), end_pos, arrow1, 2)
            self.draw_thick_aaline(screen, (255, 0, 0), end_pos, arrow2, 2)

    def draw(self, screen, camera_offset: NVector, scale: float):
        # Тут отрисовка тела, его траектории и вектора скорости
        screen_pos = (self.pos + camera_offset) * scale
        screen_x, screen_y = int(screen_pos.x), int(screen_pos.y)
        
        # Отрисовка траектории
        if len(self.trajectory) > 1:
            offset = camera_offset.array
            points = [(p + offset) * scale for p in self.trajectory]
            pygame.draw.aalines(screen, (100, 100, 100), False, points, 1)
        
        # Отрисовка тела
        scaled_radius = min(max(2, int(self.radius * scale)), 5*max(screen.get_width(), screen.get_height()))
        if (0 <= screen_x <= screen.get_width() and 
            0 <= screen_y <= screen.get_height()):
            # Основной круг
            pygame.gfxdraw.filled_circle(screen, screen_x, screen_y, scaled_radius, self.color)
            pygame.gfxdraw.aacircle(screen, screen_x, screen_y, scaled_radius, self.color)

            # Обводка для выделенного тела
            if self.selected:
                # Внешний круг со сглаживанием
                pygame.gfxdraw.aacircle(screen, screen_x, screen_y,
                                        scaled_radius + 2, (255, 255, 255))
                pygame.gfxdraw.aacircle(screen, screen_x, screen_y,
                                        scaled_radius + 1, (255, 255, 255))
        
        # Отрисовка вектора скорости
        self.draw_velocity_vector(screen, NVector(screen_x, screen_y))

    def is_clicked(self, screen_pos: NVector, click_pos: NVector, scale: float) -> bool:
        # Проверяем, попал ли клик в тело
        scaled_radius = max(CLICK_RADIUS, self.radius * scale)
        return (click_pos - screen_pos).length() <= scaled_radius

    def set_mass(self, new_mass: float) -> bool:
        # Устанавливаем новую массу тела. Возвращаем True, если масса изменилась
        new_mass = max(new_mass, MIN_MASS)
        new_mass = min(new_mass, MAX_MASS)
        if new_mass != self.mass:
                self.mass = new_mass
                self.update_radius()
                return True
        return False

    def set_velocity(self, new_velocity: NVector) -> bool:
        # Устанавливаем новую скорость тела. Возвращаем True, если скорость изменилась
        speed = new_velocity.length()
        if speed <= MAX_VELOCITY:
            if new_velocity != self.velocity:
                self.velocity = new_velocity
                return True
        elif speed > 0:
            # Если скорость превышает максимальную, нормализуем её
            self.velocity = new_velocity.normalize() * MAX_VELOCITY
            return True
        return False
