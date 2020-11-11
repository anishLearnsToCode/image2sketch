from src.image_graph.vertex import Vertex
from src.image_graph.edge import Edge


class SimpleGraph:
    def __init__(self, image_dimensions: tuple):
        # map of pixel point to a vertex {(x, y): tuple --> vertex: Vertex}
        self.vertices = {}

        self.height: int = image_dimensions[0]
        self.width: int = image_dimensions[1]
        # self.number_of_vertices: int = self.height * self.width

        for h in range(self.height):
            for w in range(self.width):
                self.add_vertex(h, w)

    # here the vertical row is represented as y and column as x
    def add_vertex(self, x, y):
        self.vertices[(x, y)] = Vertex(x, y)

    def add_edge(self, pixel_1, pixel_2):
        edge = Edge(pixel_1, pixel_2)
        self.vertices[pixel_1].add_edge(edge)
        self.vertices[pixel_2].add_edge(edge)


    # will return a list of sets of vertices that where each represent a component
    def components(self) -> list:
        result = []
        used_vertices = set()

        for position, vertex in self.vertices.items():
            if position in used_vertices:
                continue

            component = set()
            queue = []
            queue.append(vertex)

            while queue:
                current_vertex = queue.pop()
                if current_vertex.position() in used_vertices:
                    continue
                used_vertices.add(current_vertex.position())

                # adding children to the queue
                for edge in current_vertex.edges:
                    other_vertex = edge.other(current_vertex.position())
                    queue.append(self.vertices[other_vertex])

                # adding current vertex into the component
                component.add(current_vertex.position())

            # adding the component into the resulting list
            result.append(component.copy())

        return result
