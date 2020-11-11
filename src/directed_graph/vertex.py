from src.image_graph.edge import Edge


class Vertex:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.edges = set()
        self.in_degree = 0
        self.out_degree = 0

    def add_edge(self, edge: Edge):
        self.edges.add(edge)
        self.out_degree += 1

    def __str__(self):
        return str(self.degree())

    def position(self):
        return self.x, self.y

    def degree(self):
        return self.in_degree + self.out_degree
