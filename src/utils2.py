import os
import pickle

import cv2

from src.bounds import *
from src.colors import *
from src.directed_graph import DirectedGraph
from src.directories import *
from src.formats import *
from src.pencil_sketch import PencilSketch

# Constants
SQRT_2PI = np.sqrt(2 * np.pi)
SQRT_2PI_INV = 1 / SQRT_2PI
EPSILON = 10 ** -5

# Gaussian Parameters
MU = (0, 0, 0)
SIGMA_FACTOR = (4, 1.3, 1)
SIGMA = np.array(SIGMA_FACTOR) * SQRT_2PI_INV
A = (1, 1, 1)


def print_success(message: str, top_bar=True, bottom_bar=True):
    if top_bar: print('-' * 20)
    print(message)
    if bottom_bar: print('-' * 20)


def get_params_dir_name():
    return f'{MU}-{SIGMA_FACTOR}-{A}'


def make_dir_if_absent(dir: str):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def make_dirs_for_full_path_if_absent(*dirs: str):
    path = os.path.abspath('')
    for dir in dirs:
        path = os.path.join(path, dir)
        make_dir_if_absent(path)


def bounds_dir_name(bounds) -> str:
    """:return the name of the folder where the results will be saved given the particular parameters and resolution"""
    lower_bounds, upper_bounds = bounds
    return str(lower_bounds) + '_' + str(upper_bounds)


def get_frame(V, frame_number: int) -> np.ndarray:
    V.set(1, frame_number)
    return V.read()


def save_object_in_dir(obj, dir: str):
    pickle.dump(obj, open(dir, 'wb'))


def get_obj_from_dir(path: str):
    return pickle.load(open(path, 'rb'))


def inverse_gaussian(mu, sigma, Amp, y) -> np.ndarray:
    """:returns x (input) given output of gaussian function y"""
    return np.nan_to_num(sigma * np.sqrt(-2 * np.log(y * sigma * SQRT_2PI / Amp))) + mu


def deviation_vector(gaussian_inv, center_pos, surrounding_pos) -> np.ndarray:
    # when mu is not zero it should be (G_s - mu) / (G_c - mu) but in our case mu is zero so we take reduced formula
    return gaussian_inv[:, surrounding_pos[0], surrounding_pos[1]] / gaussian_inv[:, center_pos[0], center_pos[1]]


def bounded_by(ratio_vector: np.ndarray, bounds: tuple) -> np.ndarray:
    lower_bounds, upper_bounds = bounds
    lower_bounds, upper_bounds = np.array(lower_bounds), np.array(upper_bounds)
    return (ratio_vector < upper_bounds).astype(np.int) + (ratio_vector > lower_bounds).astype(np.int) == 2


def add_edges_to_lattices_based_on_deviation_spread_ratio(directed_graphs: list, vector: np.ndarray, bounds: tuple,
                                                          central_pixel: tuple, surrounding_pixel: tuple):
    allowed_bounds = bounded_by(vector, bounds)
    for directed_graph, allowed in zip(directed_graphs, allowed_bounds):
        if allowed:
            directed_graph.add_edge(central_pixel, surrounding_pixel)


def get_lattices(I: np.ndarray, bounds=BOUNDS_NORMAL):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) / 255
    gauss_inv = np.array([inverse_gaussian(mu, sigma, a, I) for mu, sigma, a in zip(MU, SIGMA, A)]) + EPSILON
    directed_graphs = [DirectedGraph(I.shape) for _ in range(3)]
    height, width = I.shape

    for row in range(height):
        for column in range(width):
            if column + 1 < width:
                right = deviation_vector(gauss_inv, center_pos=(row, column), surrounding_pos=(row, column + 1))
                add_edges_to_lattices_based_on_deviation_spread_ratio(directed_graphs, right, bounds, (row, column),
                                                                      (row, column + 1))

            if column + 1 < width and row + 1 < height:
                bottom_right = deviation_vector(gauss_inv, center_pos=(row, column),
                                                surrounding_pos=(row + 1, column + 1))
                add_edges_to_lattices_based_on_deviation_spread_ratio(directed_graphs, bottom_right, bounds,
                                                                      (row, column), (row + 1, column + 1))

            if row + 1 < height:
                bottom = deviation_vector(gauss_inv, center_pos=(row, column), surrounding_pos=(row + 1, column))
                add_edges_to_lattices_based_on_deviation_spread_ratio(directed_graphs, bottom, bounds, (row, column),
                                                                      (row + 1, column))

            if row + 1 < height and column - 1 >= 0:
                bottom_left = deviation_vector(gauss_inv, center_pos=(row, column),
                                               surrounding_pos=(row + 1, column - 1))
                add_edges_to_lattices_based_on_deviation_spread_ratio(directed_graphs, bottom_left, bounds,
                                                                      (row, column), (row + 1, column - 1))

    return directed_graphs


def get_components(lattices: list, lattices_reduction_bounds=(10, 1000)):
    """:returns the components from the lattices with lattice_reduction applied"""
    components = [lattice.components() for lattice in lattices]
    lower_bound, upper_bound = lattices_reduction_bounds
    return [lattice_reduction(component, lower_bound, upper_bound) for component in components]


def lattice_reduction(components: list, lower_bound=10, upper_bound=1000) -> list:
    return [component for component in components if lower_bound < len(component) < upper_bound]


def compute_and_save_lattices(I: np.ndarray, image_name: str, bounds=BOUNDS_NORMAL) -> None:
    image_dir = os.path.join(RESULTS_DIR, image_name, get_params_dir_name(), bounds_dir_name(bounds))
    lattices_path = os.path.join(image_dir, LATTICES_PICKLE)

    if os.path.isfile(lattices_path):
        print(f'already computed lattices for {image_name} with {bounds}')
        return None

    make_dirs_for_full_path_if_absent(RESULTS_DIR, image_name, get_params_dir_name(), bounds_dir_name(bounds))
    lattices = get_lattices(I, bounds)
    save_object_in_dir(lattices, lattices_path)
    print(f'computed lattices for {image_name} with {bounds}')


def compute_and_save_lattices_and_components(I: np.ndarray, image_name: str, bounds=BOUNDS_NORMAL):
    image_dir = os.path.join(RESULTS_DIR, image_name, get_params_dir_name(), bounds_dir_name(bounds))
    components_path = os.path.join(image_dir, COMPONENTS_PICKLE)
    lattices_path = os.path.join(image_dir, LATTICES_PICKLE)

    if os.path.isfile(components_path):
        print(f'computed components and lattices for {image_name} with {bounds}')
        return None

    make_dirs_for_full_path_if_absent(RESULTS_DIR, image_name, get_params_dir_name(), bounds_dir_name(bounds))
    compute_and_save_lattices(I, image_name, bounds)

    if not os.path.isfile(components_path):
        components = get_components(get_obj_from_dir(lattices_path), lattices_reduction_bounds=(-1, float('inf')))
        save_object_in_dir(components, components_path)
    print(f'computed components for {image_name} with {bounds}')


def compute_and_save_lattice_vertex_shaded_images(I: np.ndarray, image_name: str, bounds=BOUNDS_NORMAL):
    image_dir = os.path.join(RESULTS_DIR, image_name, get_params_dir_name(), bounds_dir_name(bounds))
    lattices_path = os.path.join(image_dir, LATTICES_PICKLE)
    vertex_dir = os.path.join(image_dir, VERTEX_COLORING)
    if os.path.isdir(vertex_dir):
        return None

    compute_and_save_lattices(I, image_name, bounds)
    lattices = get_obj_from_dir(lattices_path)
    L = [lattice_vertex_shading_image(lattice, I) for lattice in lattices]
    make_dir_if_absent(vertex_dir)
    [cv2.imwrite(os.path.join(vertex_dir, f'gaussian-{i}') + PNG, l) for i, l in enumerate(L)]


def compute_and_save_lattice_color_images(I, image_name: str, bounds=BOUNDS_NORMAL):
    compute_and_save_lattices_and_components(I, image_name, bounds)
    image_dir= os.path.join(RESULTS_DIR, image_name, get_params_dir_name(), bounds_dir_name(bounds))
    lattice_color_dir = os.path.join(image_dir, LATTICE_COLORING)
    make_dirs_for_full_path_if_absent(RESULTS_DIR, image_name, get_params_dir_name(), bounds_dir_name(bounds), LATTICE_COLORING)
    components_path = os.path.join(image_dir, COMPONENTS_PICKLE)
    frame_components = get_obj_from_dir(components_path)
    L = [lattice_image(components, I) for components in frame_components]
    [cv2.imwrite(os.path.join(lattice_color_dir, f'lattice-{i}') + PNG, l) for i, l in enumerate(L)]


def random_color() -> np.ndarray:
    return np.random.randint(0, 255, (3,))


def lattice_image(components: list, I: np.ndarray) -> np.ndarray:
    L = np.zeros(I.shape, dtype=np.uint8) + 255
    for component in components:
        pixel_color = random_color()
        for x, y in component:
            L[x, y, :] = pixel_color
    return L


def get_vertex_color(degree: int) -> np.ndarray:
    if degree == 0:
        return COLOR_9
    elif degree == 1:
        return COLOR_8
    elif degree == 2:
        return COLOR_7
    elif degree == 3:
        return COLOR_6
    elif degree == 4:
        return COLOR_5
    elif degree == 5:
        return COLOR_4
    elif degree == 6:
        return COLOR_3
    elif degree == 7:
        return COLOR_2
    elif degree == 8:
        return COLOR_1


def lattice_vertex_shading_image(lattice: DirectedGraph, I: np.ndarray):
    L = np.zeros(I.shape, dtype=np.uint8)
    for position, vertex in lattice.vertices.items():
        L[position[0], position[1], :] = get_vertex_color(vertex.degree())
    return L


def get_linear_combination(I: np.ndarray, image_name: str, weights: tuple, bounds=BOUNDS_NORMAL, brightness=30) -> tuple:
    mask_weight = 1 - sum(weights)
    S = increase_brightness(np.array(PencilSketch(I, bg_gray='').render(), dtype=np.uint8), brightness)
    vertex_dir = os.path.join(RESULTS_DIR, image_name, get_params_dir_name(), bounds_dir_name(bounds), VERTEX_COLORING)
    L = [cv2.imread(os.path.join(vertex_dir, f'gaussian-{i}') + PNG) for i in range(3)]
    return S, np.array(sum(weight * l for weight, l in zip(L, weights)) + S * mask_weight, dtype=np.uint8)


def generate_pencil_sketch_of_image(I: np.ndarray, image_name: str, weights: tuple, bounds=BOUNDS_NORMAL, brightness=30):
    compute_and_save_lattice_vertex_shaded_images(I, image_name, bounds)
    return get_linear_combination(I, image_name, weights, bounds, brightness)


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)
