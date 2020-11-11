from src.directed_graph.vertex import Vertex
from src.directed_graph.edge import Edge


class DirectedGraph:
    def __init__(self, image_dimensions: tuple):
        # map of pixel point to a vertex {(x, y): tuple --> vertex: Vertex}
        self.vertices: dict = {}
        self.height, self.width = image_dimensions

        for h in range(self.height):
            for w in range(self.width):
                self.add_vertex(h, w)

    # here the vertical row is represented as y and column as x
    def add_vertex(self, x, y):
        self.vertices[(x, y)] = Vertex(x, y)

    def add_edge(self, from_vertex: tuple, to_vertex: tuple):
        edge = Edge(from_vertex, to_vertex)
        self.vertices[from_vertex].add_edge(edge)
        self.vertices[to_vertex].in_degree += 1

    def components(self) -> list:
        """:return a list of sets of vertices that where each represent a component"""
        result = []
        used_vertices = set()

        for position, vertex in self.vertices.items():
            if position in used_vertices:
                continue

            component = set()
            queue = [vertex]

            while queue:
                current_vertex = queue.pop()
                if current_vertex.position() in used_vertices:
                    continue
                used_vertices.add(current_vertex.position())

                # adding children to the queue
                for edge in current_vertex.edges:
                    other_vertex = self.vertices[edge.to_vertex].position()
                    queue.append(self.vertices[other_vertex])

                # adding current vertex into the component
                component.add(current_vertex.position())

            # adding the component into the resulting list
            result.append(component.copy())

        return result
