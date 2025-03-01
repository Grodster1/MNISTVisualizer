import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import Nets
import pygame
import matplotlib.pyplot as plt

def maximize_data(X):
    return np.where(X > 0, 255, 0)

def load_weights(neural_net, filepath = "weights/model_weights.npz"):
    data = np.load(filepath)
    neural_net.weights_input_first = data["weights_input_first"]
    neural_net.bias_input_first = data["bias_input_first"]
    neural_net.weights_first_second = data["weights_first_second"]
    neural_net.bias_first_second = data["bias_first_second"]
    neural_net.weights_second_output = data["weights_second_output"]
    neural_net.bias_second_output = data["bias_second_output"]
    return neural_net

def save_weights(neural_net, filepath="weights/model_weights.npz"):
    np.savez(filepath,
             weights_input_first=neural_net.weights_input_first,
             bias_input_first=neural_net.bias_input_first,
             weights_first_second=neural_net.weights_first_second,
             bias_first_second=neural_net.bias_first_second,
             weights_second_output=neural_net.weights_second_output,
             bias_second_output=neural_net.bias_second_output)
    print(f"Weights and biases saved to {filepath}.")


def normalize_data(X):
    return (X / 255.0).astype(np.float32)


def split_data(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    test_size = int(m * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train = X.iloc[train_indices].to_numpy()
    X_test = X.iloc[test_indices].to_numpy()

    y_train = y.iloc[train_indices].to_numpy()
    y_test = y.iloc[test_indices].to_numpy()
    return X_train, X_test, y_train, y_test


class Board:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        pygame.display.set_caption("Neural Visualization")
        self.screen = pygame.display.set_mode((width, height))

        self.cell_size = 10
        self.grid_size = 28
        self.drawing_area = []
        self.drawing = False
        self.erasing = False
        self.drawn_grid = np.full((self.grid_size, self.grid_size), 255)

        self.last_draw_pos = None

        self.brush_sizes = [1, 2, 3]
        self.current_brush_size = 1
        self.draw_intensity = 0

        self.last_prediction = None
        self.drawing_changed = True

        self.show_grid_lines = True
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def render_board(self, neural_net):
        self.screen.fill((255, 255, 255))
        self.render_drawing_space()

        if self.drawing_changed:
            image = self.extract_drawn_image()
            if np.any(image != 0):
                image_norm = normalize_data(image)
                self.last_prediction = neural_net.predict_proba(image_norm)

            else:
                self.last_prediction = None

            self.drawing_changed = False

        self.render_output_boxes(neural_net, self.last_prediction)
        self.render_instructions()

    def render_output_boxes(self, neural_net, probabilities):
        output_size = neural_net.n_output

        square_size = self.width // (output_size + 4)
        distance = (self.width - output_size * square_size) // (output_size + 1)
        square_border_color = (0, 0, 0)
        square_border_width = 3

        total_width = output_size * square_size + (output_size - 1) * distance
        start_x = (self.width - total_width) // 2
        start_y = 100

        text_color = (0, 0, 0)

        for i in range(output_size):
            x = start_x + i * (square_size + distance)
            y = start_y

            pygame.draw.rect(
                self.screen,
                square_border_color,
                (x, y, square_size, square_size),
                width=square_border_width,
            )

            if probabilities is not None:
                fill_value = probabilities[0][i]
                fill_height = int(square_size * fill_value - square_border_width)

                if fill_value > 0.7:
                    fill_color = (0, 200, 0)
                elif fill_value > 0.3:
                    fill_color = (220, 220, 0)
                else:
                    fill_color = (200, 0, 0)

                pygame.draw.rect(
                    self.screen,
                    fill_color,
                    (x + square_border_width, y + (square_size - fill_height), square_size - 2 * square_border_width,
                     fill_height - square_border_width)
                )

                if fill_value > 0.1:
                    pct_text = f"{int(fill_value * 100)}%"
                    pct_color = (255, 255, 255) if fill_value > 0.5 else (0, 0, 0)
                    pct_surface = self.small_font.render(pct_text, True, pct_color)
                    pct_rect = pct_surface.get_rect(center=(x + square_size // 2,
                                                            y + square_size - fill_height // 2))
                    self.screen.blit(pct_surface, pct_rect)

            number_surface = self.font.render(str(i), True, text_color)
            number_rect = number_surface.get_rect(center=(x + square_size // 2, start_y - 20))
            self.screen.blit(number_surface, number_rect)

    def render_drawing_space(self):
        cell_size = self.cell_size
        grid_size = self.grid_size
        square_size = cell_size * grid_size
        outer_border_size = 3

        outer_border_color = (0, 0, 0)

        start_x = (self.width - square_size) // 2
        start_y = 250
        pygame.draw.rect(
            self.screen,
            outer_border_color,
            (start_x - outer_border_size, start_y - outer_border_size,
             square_size + 2 * outer_border_size, square_size + 2 * outer_border_size),
            width=outer_border_size)

        drawing_area = [start_x, start_y, square_size, square_size]
        self.drawing_area = drawing_area

        for i in range(grid_size):
            for j in range(grid_size):
                x = start_x + j * cell_size
                y = start_y + i * cell_size
                intensity = self.drawn_grid[i][j]
                color = (intensity, intensity, intensity)

                pygame.draw.rect(self.screen, color, (x, y, cell_size, cell_size))

                if self.show_grid_lines and intensity == 255:
                    line_color = (220, 220, 220)
                    pygame.draw.line(self.screen, line_color,
                                     (x, y + cell_size - 1),
                                     (x + cell_size - 1, y + cell_size - 1))
                    pygame.draw.line(self.screen, line_color,
                                     (x + cell_size - 1, y),
                                     (x + cell_size - 1, y + cell_size - 1))

        brush_text = f"Brush: {self.current_brush_size}"
        brush_surface = self.small_font.render(brush_text, True, (0, 0, 0))
        self.screen.blit(brush_surface, (start_x, start_y - 25))

        return drawing_area

    def render_instructions(self):
        instructions = [
            "Left click: Draw | Right click: Erase | Mouse wheel: Change brush size",
            "C: Clear | G: Toggle grid"
        ]

        y_pos = self.height - 40
        for text in instructions:
            text_surface = self.small_font.render(text, True, (80, 80, 80))
            text_rect = text_surface.get_rect(center=(self.width // 2, y_pos))
            self.screen.blit(text_surface, text_rect)
            y_pos += 20

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.drawing = True
                    self.last_draw_pos = None
                    self.draw_on_grid(event.pos)

                elif event.button == 3:
                    self.erasing = True
                    self.last_draw_pos = None
                    self.erase(event.pos)

                elif event.button == 4:
                    self.current_brush_size = min(self.current_brush_size + 1, len(self.brush_sizes))

                elif event.button == 5:
                    self.current_brush_size = max(self.current_brush_size - 1, 1)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing = False
                    self.last_draw_pos = None
                elif event.button == 3:
                    self.erasing = False
                    self.last_draw_pos = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    self.drawn_grid = np.full((self.grid_size, self.grid_size), 255)
                    self.drawing_changed = True

                elif event.key == pygame.K_g:
                    self.show_grid_lines = not self.show_grid_lines

        if self.drawing:
            mouse_pos = pygame.mouse.get_pos()
            self.draw_on_grid(mouse_pos)

        elif self.erasing:
            mouse_pos = pygame.mouse.get_pos()
            self.erase(mouse_pos)

        return True

    def draw_on_grid(self, mouse_pos):
        start_x, start_y, square_size, _ = self.drawing_area
        cell_size = self.cell_size

        rel_x, rel_y = mouse_pos[0] - start_x, mouse_pos[1] - start_y

        if 0 <= rel_x < square_size and 0 <= rel_y < square_size:
            grid_x = int(rel_x // cell_size)
            grid_y = int(rel_y // cell_size)

            if self.last_draw_pos is not None:
                last_grid_x, last_grid_y = self.last_draw_pos

                points = self.get_line_points(last_grid_x, last_grid_y, grid_x, grid_y)
                for px, py in points:
                    self.draw_at_point(px, py)
            else:
                self.draw_at_point(grid_x, grid_y)

            self.last_draw_pos = (grid_x, grid_y)
            self.drawing_changed = True

    def draw_at_point(self, grid_x, grid_y):
        brush_radius = self.brush_sizes[self.current_brush_size - 1]

        for i in range(-brush_radius, brush_radius + 1):
            for j in range(-brush_radius, brush_radius + 1):
                distance = np.sqrt(i * i + j * j)
                if distance <= brush_radius:
                    falloff = 1.0 - (distance / brush_radius) * 0.8
                    intensity = int(self.draw_intensity + (255 - self.draw_intensity) * (1 - falloff))

                    new_y, new_x = grid_y + i, grid_x + j
                    if 0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size:
                        if intensity < self.drawn_grid[new_y][new_x]:
                            self.drawn_grid[new_y][new_x] = intensity

    def erase(self, mouse_pos):
        start_x, start_y, square_size, _ = self.drawing_area
        cell_size = self.cell_size

        rel_x, rel_y = mouse_pos[0] - start_x, mouse_pos[1] - start_y

        if 0 <= rel_x < square_size and 0 <= rel_y < square_size:
            grid_x = int(rel_x // cell_size)
            grid_y = int(rel_y // cell_size)

            if self.last_draw_pos is not None:
                last_grid_x, last_grid_y = self.last_draw_pos

                points = self.get_line_points(last_grid_x, last_grid_y, grid_x, grid_y)
                for px, py in points:
                    self.erase_at_point(px, py)
            else:
                self.erase_at_point(grid_x, grid_y)

            self.last_draw_pos = (grid_x, grid_y)
            self.drawing_changed = True

    def erase_at_point(self, grid_x, grid_y):
        brush_radius = self.brush_sizes[self.current_brush_size - 1]

        for i in range(-brush_radius, brush_radius + 1):
            for j in range(-brush_radius, brush_radius + 1):
                if i * i + j * j <= brush_radius * brush_radius:
                    new_y, new_x = grid_y + i, grid_x + j
                    if 0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size:
                        self.drawn_grid[new_y][new_x] = 255

    def get_line_points(self, x0, y0, x1, y1):
        """Bresenham's line algorithm"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                if x0 == x1:
                    break
                err -= dy
                x0 += sx
            if e2 < dx:
                if y0 == y1:
                    break
                err += dx
                y0 += sy

        return points

    def extract_drawn_image(self):
        inverted_image = 255 - np.array(self.drawn_grid)
        return inverted_image.reshape(-1)


mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target'].astype(int)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_norm = normalize_data(X_train)
X_test_norm = normalize_data(X_test)

neural_net = Nets.Dense(784, 10, 128, 64)
load_weights(neural_net)
neural_net.evaluate(X_test_norm, y_test)


classes = [str(i) for i in range(10)]
confusion_matrix = confusion_matrix(y_test, neural_net.predict(X_test_norm))
display = ConfusionMatrixDisplay(confusion_matrix, display_labels=classes)
display.plot()
plt.show()


pygame.init()
running = True
board = Board(width=900, height=700)

while running:
    running = board.handle_events()
    board.render_board(neural_net)
    pygame.display.flip()
    pygame.time.delay(10)


pygame.quit()