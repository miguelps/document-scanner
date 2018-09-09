class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __str__(self):
        return "({}, {})".format(self.x, self.y)


class Line:
    def __init__(self, p0: Point, p1: Point):
        self.p0 = p0
        self.p1 = p1

        if p0.x - p1.x == 0:
            self.k = None
        else:
            self.k = (self.p0.y - self.p1.y) / (self.p0.x - self.p1.x)

        # f = ax+by+c = 0
        self.a = self.p0.y - self.p1.y
        self.b = self.p1.x - self.p0.x
        self.c = self.p0.x * self.p1.y - self.p1.x * self.p0.y

    def cross(self, line) -> Point:
        d = self.a * line.b - line.a * self.b
        if d == 0:
            return None

        x = (self.b * line.c - line.b * self.c) / d
        y = (line.a * self.c - self.a * line.c) / d

        return Point(x, y)

    def contain(self, p: Point) -> bool:
        if p is None:
            return False

        # 输入的点应该 cross(求出来的交点)
        # p 点是否落在 p0 和 p1 之间, 而不是延长线上
        if p.x > max(self.p1.x, self.p0.x):
            return False

        if p.x < min(self.p1.x, self.p0.x):
            return False

        if p.y > max(self.p1.y, self.p0.y):
            return False

        if p.y < min(self.p1.y, self.p0.y):
            return False

        return True
