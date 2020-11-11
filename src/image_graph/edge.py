# This represents an unweighted edge and stores the positions of each vertex as tuple (x, y)
class Edge:
    def __init__(self, vertex_1, vertex_2):
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2

    def other(self, position: tuple) -> tuple:
        """":returns the position tuple (x, y) of the vertex in the edge which doesn't equal position"""
        if self.vertex_1 == position:
            return self.vertex_2
        else:
            return self.vertex_1
