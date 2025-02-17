import numpy as np
import pygame
import math
from constants import *

class InputField:
    def __init__(self, x: int, y: int, width: int, height: int, label: str, 
                 min_value: float = None, max_value: float = None):
        self.held_keys = {}  # словарь для хранения зажатых клавиш и их таймеров
        print ("held_keys init")
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.value = self.previous_value = ""
        self.active = False
        self.font = pygame.font.Font(None, 20)  # уменьшенный размер шрифта
        self.label_surface = self.font.render(label, True, (255, 255, 255))
        self.min_value = min_value
        self.max_value = max_value
        self.cursor_position = 0
        self.cursor_visible = True
        self.cursor_timer = 0
        self.cursor_blink_speed = 500  # миллисекунды
        self.width = width  # сохраняем ширину поля

    def handle_key_input(self, key, unicode):
        # Обрабатываем ввод с клавиатуры
        if key == pygame.K_BACKSPACE:
            if self.cursor_position > 0:
                self.value = self.value[:self.cursor_position - 1] + self.value[self.cursor_position:]
                self.cursor_position -= 1
                return True
        elif key == pygame.K_LEFT and self.cursor_position > 0:
            self.cursor_position -= 1
            return True
        elif key == pygame.K_RIGHT and self.cursor_position < len(self.value):
            self.cursor_position += 1
            return True
        elif key in (pygame.K_UP, pygame.K_DOWN):
            try:
                value = float(self.value) if self.value not in ('', '-', 'e', '-e') else 0
                factor = 1.1 if key == pygame.K_UP else 1 / 1.1
                new_value = value * factor
                if (self.min_value is None or new_value >= self.min_value) and \
                        (self.max_value is None or new_value <= self.max_value):
                    self.value = f"{new_value:.5g}"
                    self.cursor_position = len(self.value)
                    return True
            except ValueError:
                pass
        elif unicode in '0123456789.-e':
            test_value = self.value[:self.cursor_position] + unicode + self.value[self.cursor_position:]
            # Проверяем валидность ввода
            if test_value == '-' or test_value == 'e' or test_value == '-e':
                self.value = test_value
                self.cursor_position += 1
            else:
                try:
                    # Пробуем преобразовать в число
                    if 'e' in test_value:
                        base, exp = test_value.split('e')
                        if exp in ('', '-'):  # Разрешаем вводить степень
                            valid = True
                        else:
                            float(test_value)  # Проверяем полное число
                            valid = True
                    else:
                        float(test_value)
                        valid = True
                except ValueError:
                    valid = False

                if valid:
                    # Проверяем, поместится ли текст
                    text_surface = self.font.render(test_value, True, (255, 255, 255))
                    if text_surface.get_width() < self.width - 10:
                        self.value = test_value
                        self.cursor_position += 1
                        return True
        return False
    
    def handle_event(self, event):
        # Обрабатываем события ввода. Возвращаем True если кликнули по полю или значение изменилось
        if event.type == pygame.MOUSEBUTTONDOWN:
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            if self.active and not was_active:  # Если поле только что активировали
                self.cursor_visible = True
                self.cursor_timer = pygame.time.get_ticks()
                self.previous_value = self.value
            return self.active  # Возвращаем True если кликнули по полю
            
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
                try:
                    value = float(self.value) if self.value not in ('', '-', 'e', '-e') else 0
                    if (self.min_value is None or value >= self.min_value) and \
                       (self.max_value is None or value <= self.max_value):
                        return True  # Значение изменилось и прошло валидацию
                except ValueError:
                    pass
                return False
            elif event.key == pygame.K_ESCAPE:
                self.active = False
                self.value = self.previous_value
                self.cursor_position = len(self.value)
                return False
            else:
                # Запоминаем время первого нажатия клавиши
                self.held_keys[event.key] = {
                    'time': pygame.time.get_ticks(),
                    'initial_delay': True
                }
                print('held:', self.held_keys)
                # Обрабатываем первое нажатие
                changed = self.handle_key_input(event.key, event.unicode)
                self.cursor_visible = True
                self.cursor_timer = pygame.time.get_ticks()
                return changed
        elif event.type == pygame.KEYUP and self.active:
            # Удаляем отпущенную клавишу из списка зажатых
            self.held_keys.pop(event.key, None)
            print("deleted from held_keys", event.key)
        # Обрабатываем зажатые клавиши
        current_time = pygame.time.get_ticks()
        for key, data in list(self.held_keys.items()):
            elapsed = current_time - data['time']
            print("elapsed:", elapsed)

            if data['initial_delay']:
                if elapsed >= KEY_REPEAT_DELAY:
                    data['time'] = current_time
                    data['initial_delay'] = False
                    self.handle_key_input(key, chr(key) if key < 256 else '')
            else:
                if elapsed >= KEY_REPEAT_INTERVAL:
                    data['time'] = current_time
                    self.handle_key_input(key, chr(key) if key < 256 else '')

        # Сброс таймера курсора при любом вводе
        self.cursor_visible = True
        self.cursor_timer = pygame.time.get_ticks()

        return False

    def update(self):
        # Обновление состояния курсора
        if self.active:
            current_time = pygame.time.get_ticks()
            if current_time - self.cursor_timer > self.cursor_blink_speed:
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = current_time

    def set_value(self, new_value: float):
        # Устанавливаем новое значение поля
        if np.log10(max(1, abs(new_value))) > 12:
            self.value = f"{new_value:.2e}"
        else:
            self.value = f"{new_value:.5G}"
        self.cursor_position = len(self.value)
    
    def get_value(self) -> float:
        # Возвращаем текущее значение поля как число
        try:
            return float(self.value) if self.value not in ('', '-', 'e', '-e') else 0.0
        except ValueError:
            return 0.0
    
    def draw(self, screen):
        # Отрисовка подложки
        color = (100, 100, 100) if self.active else (50, 50, 50)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 1)
        
        # Отрисовка метки
        screen.blit(self.label_surface, (self.rect.x, self.rect.y - 20))

        # Отрисовка текста без курсора
        text_surface = self.font.render(self.value, True, (255, 255, 255))
        screen.blit(text_surface, (self.rect.x + 5, self.rect.y + 5))
        self.draw_cursor(screen)

    def draw_cursor(self, screen):
        # Отрисовка курсора в нужной позиции
        if self.active and self.cursor_visible:
            # Получаем подстроку до позиции курсора
            text_before_cursor = self.value[:self.cursor_position]
            # Измеряем ширину текста до курсора
            cursor_offset = self.font.size(text_before_cursor)[0]

            # Рисуем курсор
            cursor_height = self.font.get_height()
            cursor_x = self.rect.x + 5 + cursor_offset  # 5 - отступ текста от края
            cursor_y = self.rect.y + 5  # 5 - отступ текста от края
            pygame.draw.line(screen,
                             CURSOR_COLOR,  # цвет курсора
                             (cursor_x, cursor_y),
                             (cursor_x, cursor_y + cursor_height),
                             2)  # толщина курсора


class Slider:
    def __init__(self, x: int, y: int, width: int, height: int, 
                 min_value: float, max_value: float, is_logarithmic: bool = False):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_value = min_value
        self.max_value = max_value
        self.is_logarithmic = is_logarithmic
        self.value = min_value
        self.dragging = False
    
    def handle_event(self, event) -> bool:
        # Обрабатываем события. Возвращаем True если значение изменилось
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_value(event.pos[1])
                return True
            return self.rect.collidepoint(event.pos)
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value(event.pos[1])
            return True
        
        return False
    
    def _update_value(self, y_pos: int):
        # Вычисляем положение от 0 до 1
        t = 1 - (y_pos - self.rect.top) / self.rect.height
        t = max(0, min(1, t))
        
        if self.is_logarithmic:
            # Инвертируем для масштаба
            if self.min_value < self.max_value:  # для обычных слайдеров
                self.value = self.min_value * (self.max_value/self.min_value) ** t
            else:  # для масштаба
                t = 1 - t  # инвертируем
                self.value = self.max_value * (self.min_value/self.max_value) ** t
        else:
            self.value = self.min_value + (self.max_value - self.min_value) * t

    def set_value(self, value: float):
        # Устанавливаем значение слайдера
        print('-------',value, self.value, self.min_value, self.max_value)
        self.value = max(self.min_value, min(self.max_value, value))
    
    def get_slider_position(self) -> float:
        # Возвращаем позицию ползунка в пикселях
        if self.is_logarithmic:
            if self.min_value < self.max_value:
                t = math.log(self.value/self.min_value) / math.log(self.max_value/self.min_value)
            else:
                t = 1 - math.log(self.value/self.max_value) / math.log(self.min_value/self.max_value)
        else:
            t = (self.value - self.min_value) / (self.max_value - self.min_value)
        return self.rect.top + self.rect.height * (1 - t)
    
    def draw(self, screen):
        # Отрисовка фона слайдера
        pygame.draw.rect(screen, (50, 50, 50), self.rect)
        
        # Отрисовка ползунка
        slider_pos = self.get_slider_position()
        pygame.draw.circle(screen, (200, 200, 200), 
                         (self.rect.centerx, int(slider_pos)), 8)

class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.Font(None, 24)
        self.active = False
    
    def handle_event(self, event) -> bool:
        # Обрабатываем события. Возвращаем True если кнопка была нажата
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
                return True
        return False
    
    def draw(self, screen):
        color = (255, 0, 0) if self.active else (0, 255, 0)
        pygame.draw.rect(screen, color, self.rect)
        text = "Остановить" if self.active else "Запустить"
        if self.text != "Запустить":  # для кнопки сброса
            text = self.text
        text_surface = self.font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)