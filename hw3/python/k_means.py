import random

import sys
from matplotlib.image import imread
import matplotlib.pyplot as plt


# problem 5.a
def draw_image():
    A = imread('mandrill-large.tiff')
    plt.imshow(A)
    plt.show()


class Centroid:
    def __init__(self, image_matrix):
        self.r = random.randint(0, 255)
        self.g = random.randint(0, 255)
        self.b = random.randint(0, 255)
        # a list of index of the pixel assigned to this centroid
        self.assigned_pixels = set()
        self.image_matrix = image_matrix
        (self.height, self.width, c) = image_matrix.shape

    def add_pixel(self, pixel_index):
        self.assigned_pixels.add(pixel_index)

    def remove_pixel(self, pixel_index):
        self.assigned_pixels.remove(pixel_index)

    def contains_pixel(self, pixel_index):
        return pixel_index in self.assigned_pixels

    def update_rgb(self):
        # recalculate this centroids rgb value
        pixel_count = len(self.assigned_pixels)
        if pixel_count == 0:
            return
        red_sum = green_sum = blue_sum = 0
        for index in self.assigned_pixels:
            red_sum += self.get_red_value(index)
            green_sum += self.get_green_value(index)
            blue_sum += self.get_blue_value(index)
        self.r = red_sum / pixel_count
        self.g = green_sum / pixel_count
        self.b = blue_sum / pixel_count

    def distance_from_pixel(self, pixel_index):
        return (self.get_red_value(pixel_index) - self.r) * (self.get_red_value(pixel_index) - self.r) + \
               (self.get_green_value(pixel_index) - self.r) * (self.get_green_value(pixel_index) - self.g) + \
               (self.get_blue_value(pixel_index) - self.r) * (self.get_blue_value(pixel_index) - self.b)

    def get_red_value(self, pixel_index):
        return self.image_matrix[int(pixel_index / self.width), pixel_index % self.width, 0]

    def get_green_value(self, pixel_index):
        return self.image_matrix[int(pixel_index / self.width), pixel_index % self.width, 1]

    def get_blue_value(self, pixel_index):
        return self.image_matrix[int(pixel_index / self.width), pixel_index % self.width, 2]

    def shout(self):
        print("r: " + str(self.r))
        print("g: " + str(self.g))
        print("b: " + str(self.b))


# return if this pixel is assigned to a different centroid
def assign_pixel_to_centroid(pixel_index, centroids):
    min_cost = sys.maxsize
    new_index = -1
    for i, centroid in enumerate(centroids):
        cost = centroid.distance_from_pixel(pixel_index)
        if cost < min_cost:
            min_cost = cost
            new_index = i

    # remove from previous centroid and add to current centroid
    is_updated = False
    for i, centroid in enumerate(centroids):
        contains_index = centroid.contains_pixel(pixel_index)
        if i == new_index:
            is_updated = not contains_index
            centroid.add_pixel(pixel_index)
        else:
            if contains_index:
                centroid.remove_pixel(pixel_index)

    return is_updated


# problem 5.b
def k_means(k, image_matrix):
    centroids = []
    for i in range(k):
        centroids.append(Centroid(image_matrix))
    (height, width, color) = image_matrix.shape

    pixel_updated = 1
    iterations = 0
    max_iteration = 30
    # convergence check is set to no pixel is updated
    while pixel_updated > 0 and iterations < max_iteration:
        print("pixel_updated: " + str(pixel_updated))
        iterations += 1
        pixel_updated = 0
        for i in range(height):
            for j in range(width):
                pixel_index = i * width + j
                # count # of pixel updated
                if assign_pixel_to_centroid(pixel_index, centroids):
                    pixel_updated += 1
        for centroid in centroids:
            centroid.update_rgb()
    if pixel_updated == 0:
        print("converges in " + str(iterations) + " rounds")
    else:
        print("terminates after " + str(max_iteration) + " rounds")

    return centroids


def update_picture(centroids, image_matrix):
    (height, width, c) = image_matrix.shape

    for centroid in centroids:
        for pixel_index in centroid.assigned_pixels:
            y = int(pixel_index / width)
            x = pixel_index % width
            image_matrix[x, y, 0] = centroid.r
            image_matrix[x, y, 1] = centroid.g
            image_matrix[x, y, 2] = centroid.b


# problem 5.b and 5.c
def run_k_means():
    # cluster R3 (rgb) to k=16 clusters
    # a) initialize 16 centroid of R3(0-255, 0-255, 0-255) u = int[16]
    # b) for each pixel x: repeat until no centroid moves, no new assignment to existing points
    #   i) find which cluster this pixel belongs to by minimize this cost function:
    #     J = sum((xr-ur)^2 + (xg-ug)^2 + (xb-ub)^2)
    #   ii) update each centroid u by taking the average
    #     let Xs be the pixels assigned to centroid ui
    #     for each ui
    #        ui_r = sum(Xs_r)/len(Xs)
    #        ui_g = sum(Xs_g)/len(Xs)
    #        ui_b = sum(Xs_b)/len(Xs)
    #
    #   Xs_r = sum of r value of each X in Xs

    # A = imread('mandrill-small.tiff')
    # sample outputs
    # pixel_updated: 1
    # pixel_updated: 16384
    # pixel_updated: 15263
    # pixel_updated: 8300
    # pixel_updated: 6210
    # pixel_updated: 5682
    # pixel_updated: 4929
    # pixel_updated: 8413
    # pixel_updated: 5804
    # pixel_updated: 4751
    # pixel_updated: 5533
    # pixel_updated: 6313
    # pixel_updated: 5010
    # pixel_updated: 4593
    # pixel_updated: 5795
    # pixel_updated: 6323
    # pixel_updated: 4351
    # pixel_updated: 5171
    # pixel_updated: 5584
    # pixel_updated: 5355
    # pixel_updated: 4202
    # pixel_updated: 4930
    # pixel_updated: 5971
    # pixel_updated: 5217
    # pixel_updated: 4784
    # pixel_updated: 5400
    # pixel_updated: 5999
    # pixel_updated: 4549
    # pixel_updated: 4939
    # pixel_updated: 5027
    # terminates after 30 rounds

    # A = imread('mandrill-large.tiff')
    A = imread('mandrill-large.tiff')
    A.setflags(write=1)


    centroids = k_means(16, A)



    # total = 0
    # for i, centroid in enumerate(centroids):
    #     print("centroid " + str(i))
    #     centroid.shout()
    #     total += len(centroid.assigned_pixels)

    update_picture(centroids, A)

    plt.imshow(A)
    plt.show()


def main():
    # draw_image()
    run_k_means()
    return

if __name__ == '__main__':
    main()