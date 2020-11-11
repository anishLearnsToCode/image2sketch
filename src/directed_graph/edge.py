from src.directed_graph.vertex import Vertex


# This represents an unweighted edge and stores the positions of each vertex as tuple (x, y)
class Edge:
    def __init__(self, from_vertex: tuple, to_vertex: tuple):
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex

    def other(self, position: tuple) -> tuple:
        """":returns the position tuple (x, y) of the vertex in the edge which doesn't equal position"""
        return self.to_vertex if self.from_vertex == position else self.from_vertex
