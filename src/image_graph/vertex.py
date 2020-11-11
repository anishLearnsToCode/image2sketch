from src.image_graph.edge import Edge


class Vertex:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.edges = set()
        self.degree = 0

    def add_edge(self, edge: Edge):
        self.edges.add(edge)
        self.degree += 1

    def __str__(self):
        return str(self.degree)

    def position(self):
        return (self.x, self.y)
