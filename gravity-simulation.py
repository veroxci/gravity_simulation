import concurrent.futures

import cProfile
import pstats
from operator import truediv

import pygame
import numpy as np
from util import NVector
import math
from timeit import default_timer as timer
from typing import List, Optional

from constants import *
from ui_elements import InputField, Slider, Button
from body import Body

class Simulation:
    def __init__(self): # Инициализируем основной класс программы. Создаем симуляцию с Землей и спутником на ее орбите
        pygame.init()
        # Инициализируем наш pygame


        # Создаем новое окно
        self.prev_time = timer()
        self.prev_simulation_time = self.simulation_time = 0
        self.has_bang = False
        self.need_redraw_grid = True
        pygame.display.set_caption("Cимуляция движения нескольких тел под действием сил гравитационного взаимодействия")
        # Создаем начальное состояние
        self.clock = pygame.time.Clock()
        self.bodies: List[Body] = [
            Body(NVector(0,0), 5.97e24, NVector(0,0)), # Земля
            Body(NVector(0, -9_000_000), 1000, NVector(6750, 0)), # Спутник 1 на орбите Земли
            Body(NVector(12_000_000,  0), 10000, NVector(500, -5600)), # Спутник 2 на орбите Земли
            Body(NVector(-10_000_000, -10_000_000), 1000, NVector(4000, -4000)), # Спутник 3 на орбите Земли
            Body(NVector(264_286_000, 264_286_000), 7.35e22, NVector(-730, 730)), # Луна
            Body(NVector(260_286_000, 264_286_000), 10000, NVector(-1700, 1700))
        ]
        self.selected_body: Optional[Body] = None
        self.running = True
        self.simulation_running = False
        self.time_scale = None
        self.dt = None
        self.scale = 1
        self.camera_offset = NVector()
        self.fixcamera = False
        self.rate = ACTIVE_FRAME_RATE
        
        # Сохранение состояния
        self.saved_state = None

        # Состояния взаимодействия
        self.dragging_camera = False
        self.velocity_start_pos = None
        self.mass_start_pos = None
        self.drag_start = None
        self.drag_start_offset = None

        # Интерфейс
        self.font = pygame.font.Font(None, 24)

        # Параметры камеры и масштабирования
        self.background = None
        self.grid_overlay = None
        self.screen = pygame.display.set_mode(INITIAL_WINDOW_SIZE, pygame.RESIZABLE)
        self.update_window_size(INITIAL_WINDOW_SIZE[0], INITIAL_WINDOW_SIZE[1])
        self.set_backgrgr()
        self.scale_slider.set_value(INITIAL_SCALE)
        self.handle_scale_change()
        self.time_slider.set_value(INITIAL_TIME_SCALE)
        self.handle_time_ratio_change()
        self.shown_time_ratio = self.dt * self.rate
        # Расчитываем начальное соотношение времени
        print("shown_time_ratio:", self.shown_time_ratio, self.dt)

        # Создаем один постоянный пул потоков при инициализации
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=N_WORKERS,
            thread_name_prefix='sim_worker'
        )

    def set_backgrgr(self):
        # Загружаем изображение
        try:
            self.background = pygame.image.load('images/stars.jpg')  # или .jpg
            # Масштабируем под размер окна
            self.background = pygame.transform.scale(self.background, (self.width, self.height))
        except pygame.error:
            print("Не удалось загрузить изображение")
            self.background = None

    def update_window_size(self, width: int, height: int):
        # Обновляем размеры окна и пересоздаём UI элементы
        self.width = max(MIN_WINDOW_SIZE[0], width)
        self.height = max(MIN_WINDOW_SIZE[1], height)
        self.setup_ui()

        # Масштабируем фон под новый размер
        if self.background:
            self.set_backgrgr()
        # Разрешаем перерисовку сетки
        self.need_redraw_grid = True
    
    def setup_ui(self):
        # Создаем и настраиваем элементы интерфейса
        slider_height = int(self.height * SLIDER_HEIGHT_FRACTION)
        
        # Это ползунки для масштаба и времени
        self.scale_slider = Slider(
            self.width - 40, 
            (self.height - slider_height) // 2,
            SLIDER_WIDTH, 
            slider_height,
            MIN_SCALE,
            MAX_SCALE,
            True)  # Используем логарифмическую шкалу для масштаба
        if self.scale is not None:
            self.scale_slider.set_value(self.scale)
        
        self.time_slider = Slider(
            self.width - 80,
            (self.height - slider_height) // 2,
            SLIDER_WIDTH,
            slider_height,
            MIN_TIME_SCALE,
            MAX_TIME_SCALE,
            True)  # Используем логарифмическую шкалу для времени
        if self.time_scale is not None:
            self.time_slider.set_value(self.time_scale)
        print("initial scale slider value:", self.scale_slider.value)
        
        # Кнопочки
        self.play_button = Button(
            self.width - BUTTON_WIDTH - 10,
            10,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "Запустить")
        if self.simulation_running:
            self.play_button.active = True
            
        self.reset_button = Button(
            self.width - 2 * BUTTON_WIDTH - 20,
            10,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "Сброс")

        self.fixcamera_button = Button(
            self.width - 3 * BUTTON_WIDTH - 30,
            10,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            "[камера]")

        
        # Поля ввода
        base_input_y = 60
        self.mass_input = InputField(
            10, base_input_y,
            INPUT_FIELD_WIDTH, INPUT_FIELD_HEIGHT,
            "Масса (кг)",
            MIN_MASS, MAX_MASS)

        
        self.velocity_x_input = InputField(
            10, base_input_y + INPUT_FIELD_MARGIN,
            INPUT_FIELD_WIDTH, INPUT_FIELD_HEIGHT,
            "Скорость X (м/с)",
            MIN_VELOCITY, MAX_VELOCITY)
        
        self.velocity_y_input = InputField(
            10, base_input_y + 2 * INPUT_FIELD_MARGIN,
            INPUT_FIELD_WIDTH, INPUT_FIELD_HEIGHT,
            "Скорость Y (м/с)",
            MIN_VELOCITY, MAX_VELOCITY)

    def calculate_grid_size(self) -> float:
        # Вычисляем размер сетки в зависимости от масштаба
        scale_log = math.log10(self.scale)
        scale_power = math.floor(scale_log)
        grid_size = (GRID_MIN_SIZE + GRID_MAX_SIZE)/ 2 / (10 ** scale_power)

        while grid_size * self.scale <= GRID_MIN_SIZE:
            grid_size *= 10
        while grid_size * self.scale >= GRID_MAX_SIZE:
            grid_size /= 10
        while grid_size  * self.scale <= GRID_MIN_SIZE:
            grid_size *= 2

        return grid_size

    def save_state(self):
        # Сохраняем текущее состояние системы
        self.saved_state = []
        for body in self.bodies:
            self.saved_state.append({
                'pos': NVector(body.pos),
                'velocity': NVector(body.velocity),
                'mass': body.mass,
                'color': body.color
            })

    def restore_state(self):
        # Восстанавливаем сохранённое состояние
        if self.saved_state:
            self.bodies = []
            for state in self.saved_state:
                body = Body(state['pos'], state['mass'], state['velocity'])
                body.color = state['color']
                self.bodies.append(body)
            self.simulation_running = False
            self.play_button.active = False
            self.selected_body = None


    def draw_grid(self):
        # Создаем слой сетки
        self.grid_overlay = pygame.Surface((self.width, self.height))
        self.grid_overlay.set_alpha(GRID_ALPHA)
        # Это отрисовка адаптивной координатной сетки
        grid_size = self.calculate_grid_size()
        scaled_grid = grid_size * self.scale
        
        # Вычисляем смещение сетки относительно камеры
        offset_x = self.camera_offset.x * self.scale % scaled_grid
        offset_y = self.camera_offset.y * self.scale % scaled_grid
        
        # Вычисляем начальные координаты линий сетки
        start_x = -scaled_grid + offset_x
        start_y = -scaled_grid + offset_y
        
        # Это отрисовка вертикальных линий
        x = start_x
        while x < self.width + scaled_grid:
            pygame.draw.line(self.grid_overlay, GRID_COLOR,
                           (int(x), 0), (int(x), self.height))
            x += scaled_grid
        
        # Это отрисовка горизонтальных линий
        y = start_y
        while y < self.height + scaled_grid:
            pygame.draw.line(self.grid_overlay, GRID_COLOR,
                           (0, int(y)), (self.width, int(y)))
            y += scaled_grid

    def update_selected_body_inputs(self):
        # Обновляем поля ввода при выборе тела
        if self.selected_body:
            self.mass_input.set_value(self.selected_body.mass)
            self.velocity_x_input.set_value(self.selected_body.velocity.x)
            self.velocity_y_input.set_value(self.selected_body.velocity.y)
    
    def apply_input_values(self):
        # Применяем значения из полей ввода к выбранному телу
        if self.selected_body:
            # Это обновление массы
            new_mass = self.mass_input.get_value()
            self.selected_body.set_mass(new_mass)

            # А это обновление скорости
            new_vel = NVector(
                self.velocity_x_input.get_value(),
                self.velocity_y_input.get_value()
            )
            self.selected_body.set_velocity(new_vel)

    def handle_small_fragments(self):
        # Обработка мелких осколков - удаление или слияние.
        # Вызывается после каждого столкновения и периодически в цикле симуляции.
        if len(self.bodies) <= MAX_BODIES:
            return

        # Сортируем тела по массе
        self.bodies.sort(key=lambda b: b.mass)

        # Находим все мелкие осколки (меньше 1% от масштаба системы)
        max_mass = max(b.mass for b in self.bodies)
        threshold_mass = max_mass * 0.01

        small_fragments = []
        for b in self.bodies:
            if b.mass < threshold_mass and not b.selected:  # Не удаляем выбранные тела
                small_fragments.append(b)

        # Сортируем по массе, начиная с самых мелких
        small_fragments.sort(key=lambda b: b.mass)

        # Удаляем лишние осколки
        fragments_to_remove = len(self.bodies) - MAX_BODIES
        for fragment in small_fragments[:fragments_to_remove]:
            # Находим ближайшее крупное тело для передачи импульса
            # Исключаем другие мелкие осколки из поиска
            big_bodies = [b for b in self.bodies if b.mass >= threshold_mass]
            if not big_bodies:  # Если нет крупных тел, берем любое кроме текущего
                big_bodies = [b for b in self.bodies if b != fragment]

            nearest_big = min(big_bodies,
                              key=lambda b: (b.pos - fragment.pos).length_squared())

            # Передаем импульс и массу большему телу
            nearest_big.velocity = (nearest_big.velocity * nearest_big.mass +
                                    fragment.velocity * fragment.mass) / (nearest_big.mass + fragment.mass)
            nearest_big.mass += fragment.mass
            nearest_big.update_radius()

            # Удаляем фрагмент
            self.bodies.remove(fragment)

            if len(self.bodies) <= MAX_BODIES:
                break


    def create_fragments(self, body, num_fragments, base_velocity, angle_factor, energy_factor=1.0):

        # Создает осколки тела после столкновения
        # body: исходное тело
        # num_fragments: количество осколков
        # base_velocity: базовая скорость (скорость исходного тела)
        # angle_factor: фактор разлета (0 - минимальный разлет, 1 - максимальный)
        # energy_factor: множитель кинетической энергии осколков

        fragments = []
        fragment_mass = body.mass / num_fragments
        max_angle = (math.pi / 4) * angle_factor

        for _ in range(num_fragments):
            # Случайное отклонение скорости осколка
            angle = np.random.uniform(-max_angle, max_angle)
            speed_factor = np.random.uniform(0.5, 1.5) * math.sqrt(energy_factor)

            rotated_velocity = NVector(base_velocity.x, base_velocity.y)
            rotated_velocity.rotate_rad(angle)
            fragment_velocity = rotated_velocity * speed_factor

            # Небольшой разброс в позиции осколков
            offset = NVector(np.random.uniform(-1, 1), np.random.uniform(-1, 1))
            fragment_position = body.pos + offset

            fragment = Body(fragment_position, fragment_mass, fragment_velocity)
            fragments.append(fragment)

            fragments.sort(key=lambda f: f.mass, reverse=True)
            # Самый большой осколок будет самим телом, чтобы не терять цвет и траекторию
            body.set_mass(fragments[0].mass)
            body.update_radius()
            body.pos = fragments[0].pos
            body.velocity = fragments[0].velocity
            fragments[0] = body

        return fragments

    def handle_bang(self):
        self.has_bang = False
        for body1 in self.bodies:
            for body2 in self.bodies:
                if body1 != body2 and (body1.pos - body2.pos).length() < body1.radius + body2.radius:
                    big = body1
                    small = body2
                    if small.mass > big.mass:
                        big = body2
                        small = body1
                    new_bodies_big, new_bodies_small = self.bang(big, small)
                    if len(new_bodies_small) == 0:
                        # Если при столкновении осталось одно тело - меньшее удаляем
                        self.bodies.remove(small)
                    if len(new_bodies_big) > 1:
                        # Если большое тело разбилась на осколки, добавим их.
                        # Первый осколок пропускаем - это уже существующее тело
                        self.bodies += new_bodies_big[1:]
                    if len(new_bodies_small) > 1:
                        # Аналогично для маленького
                        self.bodies += new_bodies_small[1:]
                    return

    def bang(self, big, small):
        # Вычисляем базовые параметры столкновения
        mass_ratio = small.mass / big.mass
        rel_velocity = small.velocity - big.velocity
        speed = rel_velocity.length()

        # Сначала проверяем на маленькие тела - до всех остальных проверок
        if mass_ratio < 1e-6:
            print("Определено как маленькое тело - поглощение")
            return self.merge_bodies(big, small)

        collision_angle = 0
        if speed > 0:
            direction = rel_velocity.normalize()
            centers_line = (small.pos - big.pos).normalize()
            collision_angle = abs(math.acos(direction.dot(centers_line)))
            # Берем минимальный угол к оси столкновения или к перпендикуляру
            collision_angle = min(collision_angle, math.pi - collision_angle)

            print(f"Столкновение:")
            print(f"Угол столкновения: {collision_angle * 180 / math.pi:.2f} градусов")
            print(f"Скорость: {speed:.2f} м/с")
            print(f"Отношение масс: {mass_ratio:.2e}")
            print(f"Скалярное произведение: {direction.dot(centers_line):.3f}")

        # Проверяем на скользящее столкновение - теперь смотрим на близость к нулю
        if collision_angle < 0.3:  # ~17 градусов
            print("Определено как скользящее столкновение")
            return self.elastic_collision(big, small, 0.1)

        # Стандартное столкновение
        specific_energy = (big.mass * small.mass) / (2 * (big.mass + small.mass) ** 2) * speed ** 2
        print(f"Удельная энергия: {specific_energy:.2e}")
        print(f"Порог разрушения: {BREAK_THRESHOLD:.2e}")

        if specific_energy > BREAK_THRESHOLD:
            # Проверяем возможность слияния
            escape_vel = math.sqrt(2 * G * (big.mass + small.mass) /
                                   (big.radius + small.radius))
            print(f"Скорость убегания: {escape_vel:.2f} м/с")

            if self.should_merge_fragments(big, small, speed):
                print("Решение: слияние по скорости")
                return self.merge_bodies(big, small)

            print("Решение: фрагментация")    # Распределяем массу более консервативно
            big_retained_mass_ratio = 0.85  # большое тело сохраняет 85% массы
            small_retained_mass_ratio = 0.6  # малое тело сохраняет 60% массы

            # Создаем осколки большого тела
            big_fragments = []

            # Основной осколок большого тела
            main_fragment_mass = big.mass * big_retained_mass_ratio
            remaining_big_mass = big.mass * (1 - big_retained_mass_ratio)

            big.set_mass(main_fragment_mass)
            big.is_fragment = True
            big.fragment_time = self.simulation_time
            big_fragments.append(big)

            # Второй осколок большого тела
            pos_offset = NVector(np.random.uniform(-1, 1), np.random.uniform(-1, 1))
            fragment = Body(big.pos + pos_offset * big.radius,
                            remaining_big_mass,
                            big.velocity + direction.rotate_rad(np.random.uniform(-0.5, 0.5)) * speed * 0.2)
            fragment.is_fragment = True
            fragment.fragment_time = self.simulation_time
            big_fragments.append(fragment)

            # Создаем осколки малого тела
            small_fragments = []

            # Основной осколок малого тела
            main_small_mass = small.mass * small_retained_mass_ratio
            remaining_small_mass = small.mass * (1 - small_retained_mass_ratio)

            small.set_mass(main_small_mass)
            small.is_fragment = True
            small.fragment_time = self.simulation_time
            small_fragments.append(small)

            # Второй осколок малого тела
            pos_offset = NVector(np.random.uniform(-1, 1), np.random.uniform(-1, 1))
            fragment = Body(small.pos + pos_offset * small.radius,
                            remaining_small_mass,
                            small.velocity + direction.rotate_rad(np.random.uniform(-0.5, 0.5)) * speed * 0.3)
            fragment.is_fragment = True
            fragment.fragment_time = self.simulation_time
            small_fragments.append(fragment)

            return big_fragments, small_fragments
        else:
            print("Решение: слияние по энергии")
            return self.merge_bodies(big, small)

    def merge_bodies(self, big, small):

        # Слияние двух тел с сохранением импульса

        new_mass = big.mass + small.mass
        new_velocity = (big.velocity * big.mass + small.velocity * small.mass) / new_mass
        new_position = (big.pos * big.mass + small.pos * small.mass) / new_mass

        big.pos = new_position
        big.mass = new_mass
        big.update_radius()
        big.velocity = new_velocity
        return [big], []

    def calculate_energy_transfer_ratio(self, mass_ratio):

        # Вычисляет коэффициент передачи энергии на основе соотношения масс
        # mass_ratio: отношение меньшей массы к большей
        # Возвращает коэффициент передачи энергии

        if mass_ratio <= 50000:
            return 1 - (0.1 + 0.05 * np.log10(mass_ratio / 1000))
        else:
            return 1 - 0.9 * np.exp(-(mass_ratio - 50000) / 100000)

    def should_merge_fragments(self, body1, body2, relative_speed):

        # Определяет, должны ли осколки слиться при столкновении

        # Вычисляем характерную скорость для данной пары тел
        escape_velocity = math.sqrt(2 * G * (body1.mass + body2.mass) /
                                    (body1.radius + body2.radius))

        # Если относительная скорость меньше 20% от скорости убегания - слияние
        return relative_speed < 0.2 * escape_velocity

    def elastic_collision(self, big, small, energy_loss):
        # Упругое столкновение с потерей энергии
        mass_ratio = big.mass / small.mass
        vbig = big.velocity
        vsmall = small.velocity

        # Вычисляем новые скорости с учетом потери энергии
        energy_factor = math.sqrt(1 - energy_loss)
        new_vbig = (vbig * (mass_ratio - 1) + 2 * vsmall) / (mass_ratio + 1) * energy_factor
        new_vsmall = (vsmall * (1 - mass_ratio) + 2 * mass_ratio * vbig) / (mass_ratio + 1) * energy_factor
        big.velocity = new_vbig
        small.velocity = new_vsmall

        return [big], [small]

    def calculate_acceleration_pupu(self, args):
        # Вычисляем ускорение для одного тела
        i, positions, masses = args
        pos_i = positions[i]
        acc = np.zeros(2)

        for j, pos_j in enumerate(positions):
            if i != j:
                r = pos_j - pos_i
                r_squared = np.sum(r ** 2)
                if r_squared > 0:
                    force = G * masses[i] * masses[j] / r_squared
                    acc += force / masses[i] * r / np.sqrt(r_squared)
                    if r_squared < (self.bodies[i].radius + self.bodies[j].radius) ** 2:
                        self.has_bang = True
        return acc

    def calculate_acceleration(self, args):
        i, positions, masses = args
        pos_i = positions[i]

        # Вычисляем векторы от текущего тела до всех остальных
        r = positions - pos_i  # векторы к другим телам
        r_squared = np.sum(r * r, axis=1)  # квадраты расстояний

        # Исключаем само тело и нулевые расстояния
        mask = (r_squared > 0) & (np.arange(len(positions)) != i)

        if not np.any(mask):
            return np.zeros(2)

        r_filtered = r[mask]
        r_squared_filtered = r_squared[mask][:, np.newaxis]  # добавляем размерность
        masses_filtered = masses[mask][:, np.newaxis]  # тоже добавляем размерность

        # Ускорение: G * m * r_norm / r^2
        acc = G * np.sum(masses_filtered * r_filtered / (r_squared_filtered * np.sqrt(r_squared_filtered)), axis=0)

        # Проверка столкновений
        radiuses = np.array([b.radius for b in self.bodies])[mask]
        if np.any(r_squared_filtered < (self.bodies[i].radius + radiuses) ** 2):
            self.has_bang = True

        return acc

    def calculate_derivatives(self, states: List[np.ndarray]) -> List[np.ndarray]:
        # Параллельное вычисление производных состояний
        derivatives = []
        positions = [state[:2] for state in states]
        velocities = [state[2:] for state in states]
        masses = np.array([body.mass for body in self.bodies])  # Преобразуем в numpy массив

        # Подготовка аргументов для параллельных вычислений
        args = [(i, positions, masses) for i in range(len(positions))]

        # Параллельное вычисление ускорений
        accelerations = list(self.thread_pool.map(
            self.calculate_acceleration, args))

        # Сборка производных
        for vel, acc in zip(velocities, accelerations):
            derivatives.append(np.concatenate([vel, acc]))

        return derivatives

    def rk4_step(self, dt: float):
        # Выполняем один шаг интегрирования методом Рунге-Кутты 4-го порядка
        if not self.bodies:
            return
        states = []

        # Заполняем state и заодно сбрасываем признаки осколков у тех, у кого прошло больше 100 секунд
        for body in self.bodies:
            states.append(body.get_state_vector())
            if (hasattr(body, 'is_fragment') and
                    hasattr(body, 'fragment_time') and
                    self.simulation_time - body.fragment_time > 100.0):
                body.is_fragment = False
                delattr(body, 'fragment_time')

        states = [body.get_state_vector() for body in self.bodies]

        k1_future = self.thread_pool.submit(self.calculate_derivatives, states)
        k1 = [k * dt for k in k1_future.result()]

        k2_states = [state + k * 0.5 for state, k in zip(states, k1)]
        k2_future = self.thread_pool.submit(self.calculate_derivatives, k2_states)
        k2 = [k * dt for k in k2_future.result()]

        k3_states = [state + k * 0.5 for state, k in zip(states, k2)]
        k3_future = self.thread_pool.submit(self.calculate_derivatives, k3_states)
        k3 = [k * dt for k in k3_future.result()]

        k4_states = [state + k for state, k in zip(states, k3)]
        k4_future = self.thread_pool.submit(self.calculate_derivatives, k4_states)
        k4 = [k * dt for k in k4_future.result()]

        # Обновление состояний можно также распараллелить
        def update_body(i, bodies):
            body = bodies[0]
            new_state = states[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
            body.set_state_vector(new_state)
            body.update_trajectory(self.scale)

        # Обновление состояний
        list(self.thread_pool.map(lambda args: update_body(*args),
        enumerate(zip(self.bodies, states, k1, k2, k3, k4))))

    def handle_input_events(self, event):
        # Обработка событий ввода в поля
        if self.selected_body:
            input_updated = False
            for input_field in [self.mass_input, self.velocity_x_input, self.velocity_y_input]:
                if input_field.handle_event(event):
                    input_updated = True
            if input_updated:
                self.apply_input_values()
                self.save_state()

    def handle_mouse_events(self, event):
        # Обработка событий мыши
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return
            
        mouse_pos = NVector(event.pos)
        keys = pygame.key.get_pressed()
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # ЛКМ
                # Проверяем клик по элементам интерфейса
                if self.play_button.handle_event(event):
                    self.simulation_running = self.play_button.active
                    if self.simulation_running:
                        self.rate = ACTIVE_FRAME_RATE
                    else:
                        self.rate = REST_FRAME_RATE
                    return
                
                if self.reset_button.handle_event(event):
                    self.restore_state()
                    return

                if self.fixcamera_button.handle_event(event):
                    self.change_fixcamera_state()
                    return
                
                if self.scale_slider.handle_event(event):
                    self.handle_scale_change()
                    return

                if self.time_slider.handle_event(event):
                    self.handle_time_ratio_change()
                    return
                
                # Проверяем клик по полям ввода
                for input_field in [self.mass_input, self.velocity_x_input, self.velocity_y_input]:
                    if input_field.handle_event(event):
                        return

                # Только если не кликнули по полям интерфейса, проверяем клик по телам
                clicked_body = None
                for body in self.bodies:
                    screen_pos = (body.pos + self.camera_offset) * self.scale
                    if body.is_clicked(screen_pos, mouse_pos, self.scale):
                        clicked_body = body
                        break
                
                # Обработка выбора тела
                if clicked_body:
                    for body in self.bodies:
                        body.selected = (body == clicked_body)
                    self.selected_body = clicked_body
                    self.update_selected_body_inputs()
                    
                    # Начинаем изменение параметров в зависимости от зажатых клавиш
                    if keys[pygame.K_LSHIFT]:
                        self.velocity_start_pos = clicked_body.pos
                    elif keys[pygame.K_LCTRL]:
                        self.mass_start_pos = mouse_pos
                    elif not self.simulation_running:
                        self.drag_start = mouse_pos
                        self.drag_start_offset = NVector(clicked_body.pos)
                    
                    if not self.simulation_running:
                        self.save_state()
                
                elif not self.simulation_running:
                    # Создание нового тела
                    sim_pos = (mouse_pos / self.scale) - self.camera_offset
                    new_body = Body(sim_pos, MIN_MASS, NVector(0, 0))
                    self.bodies.append(new_body)
                    
                    # Выбираем новое тело
                    for body in self.bodies:
                        body.selected = (body == new_body)
                    self.selected_body = new_body
                    self.update_selected_body_inputs()
                    self.velocity_start_pos = sim_pos
                    
                    self.save_state()
            
            elif event.button == 3:  # ПКМ
                self.dragging_camera = True
                self.drag_start = mouse_pos
                self.drag_start_offset = NVector(self.camera_offset)
            elif event.button == 4: #колесико вниз
                self.scale_slider.set_value(self.scale_slider.value * 0.8)
                self.handle_scale_change()

            elif event.button == 5: #колесико вверх
                self.scale_slider.set_value(self.scale_slider.value / 0.8)
                self.handle_scale_change()

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 or event.button == 3:  # ЛКМ/ПКМ
                self.velocity_start_pos = None
                self.mass_start_pos = None
                self.drag_start = None
                self.scale_slider.dragging = False
                self.time_slider.dragging = False
            if event.button == 3:  # ПКМ-специфик
                self.dragging_camera = False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.scale_slider.dragging:
                self.scale_slider.handle_event(event)
                self.handle_scale_change()

            elif self.time_slider.dragging:
                self.time_slider.handle_event(event)
                self.handle_time_ratio_change()
            
            elif self.dragging_camera:
                self.camera_offset = self.drag_start_offset + (mouse_pos - self.drag_start) / self.scale
            
            elif self.selected_body and not self.simulation_running:
                if self.velocity_start_pos and keys[pygame.K_LSHIFT]:
                    # Задание скорости
                    sim_pos = (mouse_pos / self.scale) - self.camera_offset
                    new_velocity = sim_pos - self.velocity_start_pos
                    #new_velocity = new_velocity * math.log(new_velocity.length())
                    if self.selected_body.set_velocity(new_velocity):
                        self.update_selected_body_inputs()
                        self.save_state()
                
                elif self.mass_start_pos and keys[pygame.K_LCTRL]:
                    # Задание массы через перетаскивание
                    delta = mouse_pos.y - self.mass_start_pos.y
                    scale_factor = math.exp(delta/3)
                    new_mass = self.selected_body.mass * scale_factor
                    if self.selected_body.set_mass(new_mass):
                        self.update_selected_body_inputs()
                        self.save_state()
                    self.mass_start_pos = mouse_pos
                
                elif self.drag_start:
                    # Перемещение тела
                    sim_pos = (mouse_pos / self.scale) - self.camera_offset
                    self.selected_body.pos = sim_pos
                    self.selected_body.clear_trajectory()
                    self.save_state()

    def handle_time_ratio_change(self):
        # При изменении положения ползунка времени обновляем шаг интегрирования
        self.time_scale = self.time_slider.value
        self.dt = INTEGRATION_TIME_FACTOR * self.time_scale

    def handle_scale_change(self):
        self.need_redraw_grid = True
        previous_scale = self.scale
        self.scale = self.scale_slider.value
        self.camera_offset = self.camera_offset - self.calculate_base_offset(previous_scale) + self.calculate_base_offset(self.scale)


    def calculate_base_offset(self, scale):
        offset_x = self.width / scale
        offset_y = self.height / scale
        return NVector(offset_x, offset_y)/2

    def handle_events(self):
        # Обработка всех событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.update_window_size(event.w, event.h)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DELETE and self.selected_body:
                    self.bodies.remove(self.selected_body)
                    self.selected_body = None
            
            self.handle_input_events(event)
            self.handle_mouse_events(event)

    def update(self):
        # Обновление состояния симуляции
        if self.simulation_running:
            dt = self.dt
            step_count = ITERATIONS_PER_FRAME
            while step_count > 0:
                self.rk4_step(dt)
                if self.has_bang:
                    self.handle_bang()
                    self.handle_small_fragments()
                self.simulation_time += dt
                step_count -= 1
            if int(self.simulation_time / dt) % 10 == 0:
                self.handle_small_fragments()  # Периодическая обработка осколков

        # Обновление мигающего курсора в полях ввода
        for input_field in [self.mass_input, self.velocity_x_input, self.velocity_y_input]:
            input_field.update()
    
    def draw(self):
        if self.selected_body and self.fixcamera:
            self.camera_offset = NVector(-1 * self.selected_body.pos.x, -1 * self.selected_body.pos.y) + self.calculate_base_offset(self.scale)
        # Отрисовка всего
        if self.background:
            self.screen.blit(self.background, (0, 0)) #Картинка
        else:
            self.screen.fill((0, 0, 0)) #Черный фон

        # Отрисовка сетки
        if self.need_redraw_grid or self.dragging_camera:
            self.draw_grid()
            self.need_redraw_grid = False
        self.screen.blit(self.grid_overlay, (0, 0))
        
        # Отрисовка всех тел
        for body in self.bodies:
            body.draw(self.screen, self.camera_offset, self.scale)
        
        # Отрисовка интерфейса
        self.scale_slider.draw(self.screen)
        self.time_slider.draw(self.screen)
        self.play_button.draw(self.screen)
        self.reset_button.draw(self.screen)
        self.fixcamera_button.draw(self.screen)

        t = timer()
        if self.running and (t - self.prev_time) > 1:
            self.shown_time_ratio = (self.simulation_time - self.prev_simulation_time)/(t - self.prev_time)
            self.prev_simulation_time = self.simulation_time
            self.prev_time = t
        # Отображение информации о масштабах
        scale_text = f"Масштаб: 1:{1/self.scale:.6G}"
        time_text = f"Время: {self.shown_time_ratio:.6G}x"
        grid_km = self.calculate_grid_size() * METERS_TO_KM
        grid_text = f"Размер клетки: {grid_km:.6G} км"
        
        scale_surface = self.font.render(scale_text, True, (255, 255, 255))
        time_surface = self.font.render(time_text, True, (255, 255, 255))
        grid_surface = self.font.render(grid_text, True, (255, 255, 255))

        self.screen.blit(scale_surface, (10, self.height - 90))
        self.screen.blit(time_surface, (10, self.height - 60))
        self.screen.blit(grid_surface, (10, self.height - 30))

        # Отрисовка полей ввода
        if self.selected_body:
            self.mass_input.draw(self.screen)
            self.velocity_x_input.draw(self.screen)
            self.velocity_y_input.draw(self.screen)
        
        # Отрисовка рамки окна
        pygame.draw.rect(self.screen, BORDER_COLOR, 
                        (0, 0, self.width, self.height), 
                        BORDER_WIDTH)
        
        pygame.display.flip()
    
    def run(self):
        # Основной цикл программы
        counter = 0
        while self.running:
            if counter % 4 == 0:
                self.handle_events()
                if self.simulation_running:
                    self.update_selected_body_inputs()
                counter = 0
            self.update()
            if counter % 4 == 0:
                self.draw()
            counter += 1
            self.clock.tick(self.rate)
        pygame.quit()

    def profiled_run(self):
        counter = 0
        # Ограничим время выполнения для профилирования
        max_iterations = 1000

        while self.running and counter < max_iterations:
            if counter % 4 == 0:
                self.handle_events()
                if self.simulation_running:
                    self.update_selected_body_inputs()
                counter = 0
            self.update()
            if counter % 4 == 0:
                self.draw()
            counter += 1
            self.clock.tick(self.rate)

    def change_fixcamera_state(self):
        if self.fixcamera:
            self.fixcamera = False
            self.fixcamera_button.text = '[]'
            self.fixcamera_button.active = False
        else:
            self.fixcamera = True
            self.fixcamera_button.text = '[камера]'
            self.fixcamera_button.active = True


if __name__ == "__main__":
    # Создаем профилировщик
    # profiler = cProfile.Profile()
    sim = Simulation()

    # Запускаем профилирование
    # profiler.enable()
    sim.run()
    # profiler.disable()

    # Анализируем результаты
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')
    # Показываем топ-20 самых затратных функций
    # stats.print_stats(20)

