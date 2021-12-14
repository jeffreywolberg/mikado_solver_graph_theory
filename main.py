import numpy as np
import matplotlib.pyplot as plt
import random
import igraph
import warnings
import heapq
import sys

plot_styles = ["-", "--", "-."]
plot_colors = ["b", "g", "r", "c", "m", "y", "k"]
plot_markers = ["v", "8", "P", "1", "d", "x", "h", "*"]

class StickHeapObj(object):
    def __init__(self, max_z, s_num):
        self.max_z = max_z
        self.s_num = s_num

    def __lt__(self, other):
        if isinstance(other, StickHeapObj):
            return self.max_z < other.max_z
        else:
            return self.max_z < other
            
    def __repr__(self):
        # multiply by -1 because heapq is a minHeap, so negating the values will produce maxHeap outcomes
        return f'Stick num: {self.s_num}, Max Z Coord: {self.max_z.round(2)*-1}'


class MikadoSolver(object):

    # ================================ Initialization =================================

    def __init__(self, num_sticks):
        self.num_sticks = num_sticks

        self.initializeSticks()
        self.setUpGraph()

    def initializeSticks(self):
        self.stick_coords = np.zeros(shape=(self.num_sticks, 2, 3))
        self.plot_details = np.empty(shape=(self.num_sticks, 3,), dtype=np.str)
        for i in range(self.num_sticks):
            self.stick_coords[i, :, 0] = random.random() * \
                5, random.random() * 5
            self.stick_coords[i, :, 1] = random.random() * \
                5, random.random() * 5
            self.stick_coords[i, :, 2] = random.random() * \
                5, random.random() * 5
            # self.stick_coords[i, :, :] *= (5/Mikado.get_dist(self.stick_coords[i]))
            self.plot_details[i, :] = [random.choice(plot_colors), random.choice(
                plot_styles), random.choice(plot_markers)]

    def setUpGraph(self):
        self.graph = igraph.Graph(directed=True)
        for i in range(self.num_sticks):
            self.graph.add_vertex(i)

        for i in range(self.num_sticks):
            for j in range(self.num_sticks):
                if self.are_sticks_overlapping(i, j):
                    z_coord, higher_stick = self.get_higher_z_coord(i, j)
                    if (higher_stick == i):
                        self.graph.add_edge(i, j)
                    else:
                        self.graph.add_edge(j, i)

    # ================================ Line methods ==========================================

    def get_line_coeffs(self, stick_num: int):
        a = self.stick_coords[stick_num, 0, 1] - \
            self.stick_coords[stick_num, 1, 1]
        b = self.stick_coords[stick_num, 1, 0] - \
            self.stick_coords[stick_num, 0, 0]
        c = self.stick_coords[stick_num, 0, 0] * self.stick_coords[stick_num, 1,
                                                                   1] - self.stick_coords[stick_num, 1, 0]*self.stick_coords[stick_num, 0, 1]
        return a, b, c

    def get_point_of_intersection(self, s_1, s_2):
        a1, b1, c1 = self.get_line_coeffs(s_1)
        a2, b2, c2 = self.get_line_coeffs(s_2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x = (b1*c2 - b2*c1)/(a1*b2-a2*b1)
            y = (c1*a2 - c2*a1) / (a1*b2 - a2*b1)
            return x, y

    def get_line_endpoints(self, s_1):
        left_x = min(self.stick_coords[s_1, :, 0].flatten())
        right_x = max(self.stick_coords[s_1, :, 0].flatten())
        bottom_y = min(self.stick_coords[s_1, :, 1].flatten())
        top_y = max(self.stick_coords[s_1, :, 1].flatten())
        return left_x, right_x, bottom_y, top_y

    def are_sticks_overlapping(self, s_1: int, s_2: int) -> bool:
        x_int, y_int = self.get_point_of_intersection(s_1, s_2)
        left_x1, right_x1, bottom_y1, top_y1 = self.get_line_endpoints(s_1)
        left_x2, right_x2, bottom_y2, top_y2 = self.get_line_endpoints(s_2)

        if x_int >= left_x1 and x_int <= right_x1 and y_int >= bottom_y1 and y_int < top_y1 and \
           x_int >= left_x2 and x_int <= right_x2 and y_int >= bottom_y2 and y_int < top_y2:
            return True
        else:
            return False

    def get_higher_z_coord(self, s_1, s_2):
        # it is assumed that the stick_nums passed in as args are overlapping in xy plane
        x_int, y_int = self.get_point_of_intersection(s_1, s_2)
        x1 = self.stick_coords[s_1, 0, 0]
        x2 = self.stick_coords[s_2, 0, 0]
        pct_along_stick1 = abs(
            x_int-x1) / abs(self.stick_coords[s_1, 1, 0] - self.stick_coords[s_1, 0, 0])
        pct_along_stick2 = abs(
            x_int-x2) / abs(self.stick_coords[s_2, 1, 0] - self.stick_coords[s_2, 0, 0])
        z_stick1 = self.stick_coords[s_1, 0, 2] + pct_along_stick1 * \
            (self.stick_coords[s_1, 1, 2] - self.stick_coords[s_1, 0, 2])
        z_stick2 = self.stick_coords[s_2, 0, 2] + pct_along_stick2 * \
            (self.stick_coords[s_2, 1, 2] - self.stick_coords[s_2, 0, 2])

        if z_stick1 >= z_stick2:
            return z_stick1, s_1
        else:
            return z_stick2, s_2

    # ================================ Visualization ==========================================

    def _draw_lines(self, draw_poi):
      for i in range(self.num_sticks):
        self.ax.plot(self.stick_coords[i, :, 0], self.stick_coords[i, :, 1], self.stick_coords[i, :, 2], label=f"Stick {i}",
            color=self.plot_details[i, 0],
            linestyle=self.plot_details[i, 1],
            marker=self.plot_details[i, 2],
            )
      for i in range(self.num_sticks):
          for j in range(self.num_sticks):
              if self.are_sticks_overlapping(i, j):
                  x, y = self.get_point_of_intersection(i, j)
                  z, higher_stick_num = self.get_higher_z_coord(i, j)
                  if draw_poi:
                      self.ax.scatter(x, y, z)

    def visualize(self, draw_poi=True):
      self.fig = plt.figure()
      self.ax = self.fig.add_subplot(111, projection='3d')
      self._draw_lines(draw_poi)
      self.ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.15),
                ncol=int(self.num_sticks/3), fancybox=True, shadow=True)
      plt.show()

    # ================================ Graph Algorithms ==========================================

    def remove_sticks(self, visualize_after_each=False):
        max_heap = []
        heapq.heapify(max_heap)
        removed_sticks = []
        adj_matrix = self.graph.get_adjacency()
        sticks = np.ones(shape=(self.num_sticks,)) * 2

        for i in range(self.num_sticks):
            # see which sticks don't have sticks above them
            if all(adj_matrix[:, i] != sticks):
                print(f"Inserting new stick {i} into heap {max_heap}")
                # multiply by -1 because heapq is a minHeap, so negating the values will produce maxHeap outcomes
                heapq.heappush(max_heap, StickHeapObj(
                    max(self.stick_coords[i, :, 2]) * -1, i))

        while(len(max_heap) > 0):
            # remove the stick and visualize the removal
            stick_num = heapq.heappop(max_heap).s_num
            print("After extracting max:", max_heap)
            removed_sticks.append(stick_num)
            self.stick_coords[stick_num, :, 0] = 7
            if visualize_after_each:
                self.visualize(draw_poi=True)

            # check all the sticks below the removed stick
            sticks_below = adj_matrix[stick_num, :]
            for i, val in enumerate(sticks_below):
                # if a stick was laying below the removed stick, delete the edge in graph
                # (the deletion signifies that the removed stick is no longer on top of it)
                if val == 2:
                    self.graph.delete_edges((stick_num, i))
                    adj_matrix = self.graph.get_adjacency()
                    # determine if the removal of this stick caused any other sticks to be able to be picked up
                    if all(adj_matrix[:, i] != sticks):
                        # multiply by -1 because heapq is a minHeap, so negating the values will produce maxHeap outcomes
                        heapq.heappush(max_heap, StickHeapObj(
                            max(self.stick_coords[i, :, 2]) * -1, i))
                        print(f"Inserting stick {i} into heap: {max_heap}")
                        

        print("Removed sticks:", removed_sticks)

if __name__ == "__main__":
  def test(event):
    print(event.key)
  num_sticks = 10
  obj = MikadoSolver(num_sticks)
  obj.visualize(draw_poi=True)
  obj.remove_sticks(visualize_after_each=True)

