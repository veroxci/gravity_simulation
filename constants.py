# Размеры окна и интерфейса
BORDER_COLOR = (192, 192, 192)  # серебристый цвет рамки
BORDER_WIDTH = 2
SLIDER_HEIGHT_FRACTION = 0.85  # ползунки занимают 85% высоты
INITIAL_WINDOW_SIZE = (1024, 768)
MIN_WINDOW_SIZE = (640, 480)
WINDOW_MARGIN = 10  # минимальное расстояние от объектов до края окна

# Сетка
GRID_ALPHA = 160
GRID_COLOR = (64, 64, 64, GRID_ALPHA)

# Физические константы
G = 6.67430e-11  # гравитационная постоянная
MIN_MASS = 1000
MAX_MASS = 1e100
MIN_VELOCITY = -300_000_000
MAX_VELOCITY = 300_000_000
REL_EFFECTS_THRESHOLD = 42_000_000

METERS_TO_KM = 0.001  # для перевода метров в километры

# Визуализация
ACTIVE_FRAME_RATE = 400
REST_FRAME_RATE = 30
ITERATIONS_PER_FRAME = 10
N_WORKERS = 10
TRAJECTORY_LENGTH = 1000
MIN_SCALE = 1e-16  # для объектов размером в десятки световых лет
MAX_SCALE = 0.0001   # для объектов размером в десятки километров
INITIAL_SCALE = 0.00002
MIN_TIME_SCALE = 0.01
MAX_TIME_SCALE = 1e9
INITIAL_TIME_SCALE = 200000
CLICK_RADIUS = 30  # область выбора объекта
MAX_OBJECT_SCREEN_FRACTION = 0.25
MIN_BODY_RADIUS = 5
GRID_MIN_SIZE = 30  # минимальный размер сетки в пикселях
GRID_MAX_SIZE = 80 # максимальный размер сетки в пикселях


# Интерфейс
INPUT_FIELD_WIDTH = 120
INPUT_FIELD_HEIGHT = 30
INPUT_FIELD_MARGIN = 60
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 30
SLIDER_WIDTH = 20
CURSOR_COLOR = (255, 255, 255)
KEY_REPEAT_INTERVAL = 50  # миллисекунды между повторениями
KEY_REPEAT_DELAY = 200  # миллисекунды до начала повторения



# Физика
INTEGRATION_TIME_FACTOR = 0.0001  # множитель для шага интегрирования

# Пороговые значения для разных сценариев при столкновении
BREAK_THRESHOLD = 1e4  # Порог для разрушения

# Максимальное количество тел в симуляции
MAX_BODIES = 50

