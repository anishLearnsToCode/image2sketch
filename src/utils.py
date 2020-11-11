import cv2
import numpy as np
from math import pi
from src.image_graph.simple_graph import SimpleGraph
import os
import pickle
import concurrent.futures
from src.pencil_sketch import PencilSketch
from src.control_parameters import *


# constants
SQRT_2PI = np.sqrt(2 * pi)
SQRT_2PI_INV = 1 / SQRT_2PI
EPSILON = 10 ** -5

# Colors
COLOR_1 = np.array([255, 255, 255])
COLOR_2 = np.array([229, 232, 232])
COLOR_3 = np.array([204, 209, 209])
COLOR_4 = np.array([178, 186, 187])
COLOR_5 = np.array([153, 163, 164])
COLOR_6 = np.array([127, 140, 141])
COLOR_7 = np.array([97, 106, 107])
COLOR_8 = np.array([81, 90, 90])
COLOR_9 = np.array([66, 73, 73])
COLOR_BLUE = np.array([255, 0, 0])

# Formats
JPG = '.jpg'
JPEG = '.jpeg'
PNG = '.png'

# BOUNDS
BOUNDS_NORMAL = ((0.86, 0.94, 0.94), (1.162, 1.063, 1.063))
BOUNDS_1 = ((0.9, 0.92, 0.94), (1.111, 1.086, 1.063))
BOUNDS_2 = ((0.92, 0.92, 0.94), (1.086, 1.086, 1.063))
BOUNDS_3 = ((0.94, 0.92, 0.94), (1.063, 1.086, 1.063))
BOUNDS_4 = ((0.86, 0.90, 0.94), (1.162, 1.111, 1.063))
BOUNDS_5 = ((0.86, 0.88, 0.94), (1.162, 1.136, 1.063))
BOUNDS_6 = ((0.86, 0.88, 0.92), (1.162, 1.136, 1.087))
BOUNDS_7 = ((0.98, 0.98, 0.98), (1.02, 1.02, 1.02))

# Important Directories
RESULTS_DIR = os.path.abspath('../results')
DEVIATIONS = 'deviations'
DEVIATIONS_PICKLE = 'deviations.p'
LATTICES_PICKLE = 'lattices.p'
COMPONENTS_PICKLE = 'components.p'
LATTICE_COLORING = 'lattice-coloring'
VERTEX_COLORING = 'vertex-coloring'

# Gaussian Parameters
MU = (0, 0, 0)
SIGMA_FACTOR = (4, 1.3, 1)
SIGMA = np.array(SIGMA_FACTOR) * SQRT_2PI_INV
A = (1, 1, 1)


def show_video(video_stream):
    while True:
        ret, frame = video_stream.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def show_image(I, title='image'):
    cv2.imshow(title, I)
    cv2.waitKey(0)


def fps(video) -> int:
    return video.get(cv2.CAP_PROP_FPS)


def dimensions(video):
    return int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


def frame_count(video) -> int:
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


def frame_width(video) -> int:
    return int(video.get(cv2.CAP_PROP_FRAME_WIDTH))


def frame_height(video) -> int:
    return int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


def gray_scale_video_frames(video):
    gray_video = np.zeros((frame_count(video), frame_height(video), frame_width(video)))
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            gray_video[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            i += 1
        else:
            break

    return gray_video


def inverse_gaussian(mu, sigma, A, y) -> np.ndarray:
    """:returns x (input) given output of gaussian function y"""
    return np.nan_to_num(sigma * np.sqrt(-2 * np.log(y * sigma * SQRT_2PI / A)) + mu, 0)
    # return sigma * np.sqrt(-2 * np.log(y * sigma * SQRT_2PI / A)) + mu


def deviation_vector(gaussian_inv, center_pos, surrounding_pos) -> np.ndarray:
    # when mu is not zero it should be (G_s - mu) / (G_c - mu) but in our case mu is zero so we take reduced formula
    return gaussian_inv[:, surrounding_pos[0], surrounding_pos[1]] / gaussian_inv[:, center_pos[0], center_pos[1]]


def deviation_vectors(frame) -> tuple:
    # converting frame to grayscale
    I = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # normalizing frame
    I = I / 255

    # creating the corresponding inverse gaussian matrices to compute deviation spread
    gauss_inv = np.zeros((3, I.shape[0], I.shape[1]))

    for i in range(3):
        gauss_inv[i] = inverse_gaussian(MU[i], SIGMA[i], A[i], I)

    # Adding a Scaling Constant to the Inverse Gaussians to avoid 0/0 division
    gauss_inv = gauss_inv + EPSILON

    # Moving widows across the Image to create deviation spread vector
    m = 1
    w = 2 * m + 1
    height, width = I.shape
    deviation_spreads = {}

    # We iterate over all possible windows
    for row in range(m, height - m):
        for column in range(m, width - m):

            # first row of window
            for i in range(column - m, column + m + 1):
                deviation_spreads[((row, column), (row - m, i))] = deviation_vector(gauss_inv,
                                                                                    center_pos=(row, column),
                                                                                    surrounding_pos=(row - m, i))

            # last column of window
            for i in range(row - m + 1, row + m + 1):
                deviation_spreads[((row, column), (i, column + m))] = deviation_vector(gauss_inv,
                                                                                       center_pos=(row, column),
                                                                                       surrounding_pos=(i, column + m))

            # last row of window
            for i in range(column - m, column + m):
                deviation_spreads[((row, column), (row + m, i))] = deviation_vector(gauss_inv,
                                                                                    center_pos=(row, column),
                                                                                    surrounding_pos=(row + m, i))

            # first column of window
            for i in range(row - m + 1, row + m):
                deviation_spreads[((row, column), (i, column - m))] = deviation_vector(gauss_inv,
                                                                                       center_pos=(row, column),
                                                                                       surrounding_pos=(i, column - m))

    return deviation_spreads, gauss_inv


def create_lattice_with_deviation_vectors(I, deviation_spreads, lower_bound, upper_bound, gaussian: int):
    # creating a set that maintains memory so as to not compare duplicate edges
    memory = set()
    lattice = SimpleGraph(I.shape)

    for (center_pixel, surrounding_pixel), vector in deviation_spreads.items():
        if (center_pixel, surrounding_pixel) in memory or (surrounding_pixel, center_pixel) in memory:
            continue

        memory.add((center_pixel, surrounding_pixel))

        val = False
        ratio = vector[gaussian]
        if not np.isnan(ratio) and (lower_bound[gaussian] <= ratio <= upper_bound[gaussian]):
            val = True

        if val:
            lattice.add_edge(center_pixel, surrounding_pixel)

    return lattice


def create_lattices_with_deviation_vectors(I, deviation_spreads, bounds=BOUNDS_NORMAL):
    lower_bounds, upper_bounds = bounds
    lattices = [0, 0, 0]

    for i, lattice in enumerate(lattices):
        lattices[i] = create_lattice_with_deviation_vectors(I, deviation_spreads, lower_bounds, upper_bounds, gaussian=i)

    return lattices


def random_color() -> np.ndarray:
    return np.random.randint(0, 255, (3,))


def lattice_image(components, I) -> np.ndarray:
    L = np.zeros(I.shape, dtype=np.uint8)

    # sorted_components = sorted(components, key=lambda x:len(x))

    for component in components:
        pixel_color = random_color()
        # setting every pixel in this component to that random color
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


def lattice_vertex_degree_repr(lattice, I) -> np.ndarray:
    L = np.zeros(I.shape, dtype=np.uint8)

    # we will iterate over each vertex.pixel in teh Lattice and mark all pixels of the same degree as the same color
    for position, vertex in lattice.vertices.items():
        L[position[0], position[1], :] = get_vertex_color(vertex.degree)

    return L


def bounds_dir_name(bounds) -> str:
    """:return the name of the folder where the results will be saved given the particular parameters and resolution"""
    lower_bounds, upper_bounds = bounds
    return str(lower_bounds) + '_' + str(upper_bounds)


def frame_dir_name(frame_number: int) -> str:
    return 'frame-' + str(frame_number)


def make_dir_if_absent(path: str) -> None:
    if not os.path.isdir(path): os.mkdir(path)


def get_frame(V, frame_count: int) -> np.ndarray:
    V.set(1, frame_count)
    return V.read()


def get_params_dir_name():
    return f'{MU}-{SIGMA_FACTOR}-{A}'


def save_lattices_for_video(V, bounds=((0.86, 0.94, 0.94), (1.162, 1.063, 1.063)), resolution=10, video_name='video'):
    """:return None. this will save the frames of the video with the corresponding lattice and also the lattice
    images """
    # important directories
    video_dir = os.path.join(RESULTS_DIR, video_name)
    params_dir = os.path.join(video_dir, get_params_dir_name())
    deviation_dir = os.path.join(params_dir, DEVIATIONS)
    bounds_dir = os.path.join(params_dir, bounds_dir_name(bounds))

    print('creating lattice frames for:', video_name)

    make_dir_if_absent(RESULTS_DIR)
    make_dir_if_absent(video_dir)
    make_dir_if_absent(params_dir)
    make_dir_if_absent(bounds_dir)
    make_dir_if_absent(deviation_dir)

    number_of_frames = frame_count(V)

    for frame_number in range(0, number_of_frames, resolution):
        # check whether this frame has been computed or not
        frame_dir = os.path.join(bounds_dir, frame_dir_name(frame_number))

        if os.path.isdir(frame_dir):
            continue

        make_dir_if_absent(frame_dir)

        # create lattice images from lattice
        lattice_dir = os.path.join(frame_dir, LATTICES_PICKLE)
        print(f'computing lattices for frame: {frame_number}')
        if os.path.exists(lattice_dir):
            lattices = get_obj_from_dir(lattice_dir)
        else:
            # obtaining the frame
            _, frame = get_frame(V, frame_number)

            # obtaining the deviation vectors
            print('computing deviations for frame:', frame_number)
            deviations_file_path = os.path.join(deviation_dir, 'frame-' + str(frame_number) + '-deviations.p')
            if os.path.exists(deviations_file_path):
                deviations = get_obj_from_dir(deviations_file_path)
            else:
                # computing the deviation vectors
                deviations, _ = deviation_vectors(frame)

                # saving the deviation vectors
                save_object_in_dir(deviations, deviations_file_path)
            print(f'deviation vectors loaded for frame: {frame_number}')

            # computing the lattices for the frame
            lattices = create_lattices_with_deviation_vectors(frame, deviations, bounds)

            # saving the lattices for this frame
            pickle.dump(lattices, open(lattice_dir, 'wb'))
            print(f'loaded the lattices for frame: {frame_number}')

            # computing the 3 lattice vector coloring vectors
            L = [0, 0, 0]
            for i, lattice in enumerate(lattices):
                L[i] = lattice_vertex_degree_repr(lattice, frame)

            # saving the 3 lattice vectors coloring images
            for i, l in enumerate(L):
                cv2.imwrite(os.path.join(frame_dir, 'gaussian-' + str(i)) + JPEG, l)
            print(f'saved lattice images for: {frame_number}')
            print('---------------------------------------')
    print('Task completed for:', video_name)


def play_video_with_lattice(V, video_name: str, bounds=BOUNDS_NORMAL,
                            delay=100):
    """Plays the Video with given bounds and name from the results directory"""
    lower_bounds, upper_bounds = bounds
    number_of_frames = frame_count(V)
    video_dir = os.path.join(RESULTS_DIR, video_name)
    params_dir = os.path.join(video_dir, get_params_dir_name())
    bounds_dir = os.path.join(params_dir, bounds_dir_name(bounds))

    for frame_number in range(0, number_of_frames):
        frame_dir = os.path.join(bounds_dir, frame_dir_name(frame_number))

        if os.path.isdir(frame_dir):
            _, frame = get_frame(V, frame_number)
            L1 = cv2.imread(frame_dir + '/gaussian-0.jpeg')
            L2 = cv2.imread(frame_dir + '/gaussian-1.jpeg')
            L3 = cv2.imread(frame_dir + '/gaussian-2.jpeg')
            cv2.imshow('Original Video', frame)
            cv2.imshow('gaussian-1', L1)
            cv2.imshow('gaussian-2', L2)
            cv2.imshow('gaussian-3', L3)
            cv2.waitKey(delay)


def play_video_with_specific_lattice(V, video_name: str, bounds=(BOUNDS_NORMAL, ), delay=100, lattice=2, resolution=10):
    """Plays the Video with given bounds and name from the results directory"""
    number_of_frames = frame_count(V)
    video_dir = os.path.join(RESULTS_DIR, video_name)
    params_dir = os.path.join(video_dir, get_params_dir_name())
    print(f'playing frames from {params_dir}')
    bounds_dirs = [os.path.join(params_dir, bounds_dir_name(bound)) for bound in bounds]

    for frame_number in range(0, number_of_frames, resolution):
        frame_dirs = [os.path.join(bounds_dir, frame_dir_name(frame_number)) for bounds_dir in bounds_dirs]
        _, frame = get_frame(V, frame_number)
        L = [cv2.imread(frame_dir + '/gaussian-' + str(lattice) + JPEG) for frame_dir in frame_dirs if os.path.isdir(frame_dir)]
        cv2.imshow('Original Video', frame)
        [cv2.imshow(f'gaussian-{i}', image) for i, image in enumerate(L)]
        cv2.waitKey(delay)


def relation_between_components(components_1, components_2):
    results = {}
    for i, component_1 in enumerate(components_1):
        relations = []
        for j, component_2 in enumerate(components_2):
            intersection = component_1.intersection(component_2)
            relations.append((j, len(intersection) / len(component_1)))
        results[i] = relations
    return results


def relation_between_lattices(components_1, components_2):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [executor.submit(relation_between_components, components_1[i], components_2[i]) for i in range(3)]
        return [task.result() for task in tasks]


def get_lattices(frame_dir: str):
    lattices_path = os.path.join(frame_dir, LATTICES_PICKLE)
    return get_obj_from_dir(lattices_path)


def get_components(frame_dir: str):
    components_path = os.path.join(frame_dir, COMPONENTS_PICKLE)
    if os.path.isfile(components_path):
        return get_obj_from_dir(components_path)
    else:
        lattices = get_lattices(frame_dir)
        components = [lattice.components() for lattice in lattices]
        save_object_in_dir(components, components_path)
        return components


def save_object_in_dir(obj, dir: str):
    pickle.dump(obj, open(dir, 'wb'))


def get_obj_from_dir(path: str):
    return pickle.load(open(path, 'rb'))


def get_relationship_path(frame_number_1: int, frame_number_2: int):
    return f'relationship-{frame_number_1}-{frame_number_2}.p'


def relation_between_2_frames(video_name: str, bounds: tuple, frame_number_1: int, frame_number_2: int) -> None:
    print(f'creating the relationship between {frame_number_1} and {frame_number_2}')
    video_dir = os.path.join(RESULTS_DIR, video_name, get_params_dir_name(), bounds_dir_name(bounds))
    frame_1_dir = os.path.join(video_dir, frame_dir_name(frame_number_1))
    frame_2_dir = os.path.join(video_dir, frame_dir_name(frame_number_2))
    relation_results_path = os.path.join(frame_1_dir, get_relationship_path(frame_number_1, frame_number_2))

    if os.path.isfile(relation_results_path):
        return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        t1 = executor.submit(get_components, frame_1_dir)
        t2 = executor.submit(get_components, frame_2_dir)
        components_1 = t1.result()
        components_2 = t2.result()

    results = relation_between_lattices(components_1, components_2)
    save_object_in_dir(results, relation_results_path)
    print(f'created the relationship for frame {frame_number_1} and {frame_number_2}')


def create_relationship_between_frames_async(V, video_name: str, bounds: tuple, resolution=10):
    lower_bounds, upper_bounds = bounds
    video_dir = os.path.join(RESULTS_DIR, video_name, bounds_dir_name(lower_bounds, upper_bounds))
    number_of_frames = frame_count(V)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(0, number_of_frames, resolution):
            frame_dir_1 = os.path.join(video_dir, frame_dir_name(i))
            frame_dir_2 = os.path.join(video_dir, frame_dir_name(i + resolution))
            if os.path.isdir(frame_dir_1) and os.path.isdir(frame_dir_2):
                executor.submit(relation_between_2_frames, video_name, bounds, i, i + resolution)
    print('-------------------------')
    print(f'Relationships created for Video: {video_name}')
    print('-------------------------')


def create_relationship_between_frames_sync(V, video_name: str, bounds: tuple, resolution=10):
    video_dir = os.path.join(RESULTS_DIR, video_name, get_params_dir_name(), bounds_dir_name(bounds))
    number_of_frames = frame_count(V)
    for i in range(0, number_of_frames, resolution):
        frame_dir_1 = os.path.join(video_dir, frame_dir_name(i))
        frame_dir_2 = os.path.join(video_dir, frame_dir_name(i + resolution))
        if os.path.isdir(frame_dir_1) and os.path.isdir(frame_dir_2):
            relation_between_2_frames(video_name, bounds, i, i + resolution)
    print('-------------------------')
    print(f'Relationships created for Video: {video_name}')
    print('-------------------------')


def color_component_in_image(I, component: set, color=COLOR_BLUE) -> None:
    for x, y in component:
        I[x, y, :] = color


def get_next_component(relationship: dict, component_no: int) -> int:
    mappings = relationship[component_no]
    return sorted(mappings, key=lambda x: x[1])[-1][0]


def get_next_components(relationship: list, component_nos: tuple) -> tuple:
    return tuple([get_next_component(relationship[i], component_nos[i]) for i in range(3)])


def play_video_with_lattice_relationship(V, video_name: str, bounds: tuple, component_nos: tuple, resolution=10,
                                         delay=100):
    lower_bounds, upper_bounds = bounds
    video_dir = os.path.join(RESULTS_DIR, video_name, bounds_dir_name(lower_bounds, upper_bounds))
    number_of_frames = frame_count(V)

    for frame_number in range(0, number_of_frames, resolution):
        frame_dir = os.path.join(video_dir, frame_dir_name(frame_number))

        if os.path.isdir(frame_dir):
            components = get_components(frame_dir)
            _, frame = get_frame(V, frame_number)
            L = [cv2.imread(os.path.join(frame_dir, f'gaussian-{i}.jpeg')) for i in range(3)]
            specific_components = [components[i][component_nos[i]] for i in range(3)]
            [color_component_in_image(L[i], specific_components[i]) for i in range(3)]
            cv2.imshow('Original Video', frame)
            [cv2.imshow(f'gaussian-{i}', L[i]) for i in range(3)]
            cv2.waitKey(delay)
            relationship = get_obj_from_dir(
                os.path.join(frame_dir, get_relationship_path(frame_number, frame_number + resolution)))
            component_nos = get_next_components(relationship, component_nos)


def save_video_frames(V, video_name: str, start_frame=0, resolution=10):
    video_dir = os.path.join(RESULTS_DIR, video_name)
    frames_dir = os.path.join(video_dir, 'frames')
    make_dir_if_absent(RESULTS_DIR)
    make_dir_if_absent(video_dir)
    make_dir_if_absent(frames_dir)
    number_of_frames = frame_count(V)
    for frame_number in range(start_frame, number_of_frames, resolution):
        frame_path = os.path.join(frames_dir, frame_dir_name(frame_number)) + JPG
        if os.path.isfile(frame_path):
            continue
        _, I = get_frame(V, frame_number)
        cv2.imwrite(frame_path, I)
        print(f'wrote frame {frame_number} for {video_name}')
    print('--------------------------')
    print(f'completed frame generation for {video_name}')
    print('--------------------------')


def get_deviations(I: np.ndarray, deviation_path: str) -> np.ndarray:
    if os.path.exists(deviation_path):
        return get_obj_from_dir(deviation_path)
    else:
        deviations, _ = deviation_vectors(I)
        save_object_in_dir(deviations, deviation_path)
        return deviations


def create_vertex_shaded_image(I: np.ndarray, image_name: str, bounds=BOUNDS_NORMAL):
    image_dir = os.path.join(RESULTS_DIR, image_name)
    params_dir = os.path.join(image_dir, get_params_dir_name())
    bounds_dir = os.path.join(params_dir, bounds_dir_name(bounds))
    vertices_coloring_dir = os.path.join(bounds_dir, VERTEX_COLORING)
    deviations_path = os.path.join(params_dir, DEVIATIONS_PICKLE)
    lattices_path = os.path.join(bounds_dir, LATTICES_PICKLE)
    make_dir_if_absent(image_dir)
    make_dir_if_absent(params_dir)
    make_dir_if_absent(bounds_dir)

    if os.path.isdir(vertices_coloring_dir):
        print(f'lattice images already created for {image_name}')
        return None

    make_dir_if_absent(vertices_coloring_dir)

    # loading in the lattices
    print(f'loading lattices for {image_name}')
    if os.path.exists(lattices_path):
        lattices = get_obj_from_dir(lattices_path)
    else:
        # loading in the deviations and creating lattices from that
        print(f'loading in the deviations for {image_name}')
        deviations = get_deviations(I, deviations_path)
        print(f'deviations loaded for {image_name}')
        lattices = create_lattices_with_deviation_vectors(I, deviations, bounds=bounds)
        save_object_in_dir(lattices, lattices_path)
    print(f'lattices loaded for {image_name}')

    # creating vertex shading images and saving them
    L = [lattice_vertex_degree_repr(lattice, I) for lattice in lattices]
    [cv2.imwrite(os.path.join(vertices_coloring_dir, f'lattice-{i}') + JPG, l) for i, l in enumerate(L)]


def get_linear_combination(I: np.ndarray, image_name: str, weights: tuple, bounds=BOUNDS_NORMAL) -> np.ndarray:
    mask_weight = 1 - sum(weights)
    image_dir = os.path.join(RESULTS_DIR, image_name, get_params_dir_name(), bounds_dir_name(bounds), VERTEX_COLORING)
    L = [np.array(cv2.imread(os.path.join(image_dir, f'lattice-{i}') + JPG), dtype=np.int) for i in range(3)]
    S = np.array(PencilSketch(I, bg_gray='').render(), dtype=np.int)
    return np.array(sum(weight * l for weight, l in zip(L, weights)) + S * mask_weight, dtype=np.uint8)
