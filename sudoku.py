from numpy_ringbuffer import RingBuffer
from fuzzywuzzy import process
import numpy as np
import cv2

from tools import Singleton
from neural_model import NeuralModel
import solve_sudoku


@Singleton
class Sudoku:
    def __init__(self):
        size = (9, 9)
        self.already_solved = {}
        self.already_solved_numbers = {}
        self.already_solved_false = []
        self.grid = np.empty(size, dtype=np.object) #Array of objects of class grid
        for i in range(size[0]):
            for j in range(size[1]):
                self.grid[i,j] = Grid()

    def update_grid(self, image, grid_position, physical_position):
        self.grid[grid_position].update(image, grid_position, physical_position)

#     def guess_sudoku(self, confidence_threshold=0):
    def predict_grid_num(self, confidence_threshold=0):
        for i in range(9):
            for j in range(9):
                grid = self.grid[i,j] 
                num = grid.guess_number(confidence_threshold=confidence_threshold) #calling the class_method of grid class

        
    def write_solution(self, sudoku_image, solution, ignore=None , required_num_in_sol=""):
        if solution is not False:
            cols   = '123456789'
            rows   = 'ABCDEFGHI'
            for i in range(9):
                for j in range(9):
                    number = solution[rows[i] + cols[j]]
                    grid = self.grid[i,j]
                    if(str(number) in required_num_in_sol):
                        if ignore is None:
                            if grid.number == 0:

                                grid.write(sudoku_image, number)
                        else:
                            if (i, j) not in ignore:
                                grid.write(sudoku_image, number)


    def get_existing_numbers(self):
        existing_numbers = []
        for i in range(9):
            for j in range(9):
                grid = self.grid[i,j]
                if grid.number != 0:
                    existing_numbers.append((i,j))

        return existing_numbers

    def as_string(self):
        'Turns the numbers of the sudoku into a string to be read by algorithm'
        # 0:00:00.000064
        string = ''

        array = np.ravel(self.grid)
        for guy in array:
            string += str(guy.number)

        return string


    def solve_by_approximate(self, approximate=False):
        'If it finds a sudoku similar to one it has already done, uses its solution'
        'thus it uses past history of the solved sequeces in the current session of sudoku solutions '
        string = self.as_string()
        if string in self.already_solved.keys():
            return self.already_solved[string], self.already_solved_numbers[string]
        else:
            # We save the attempts that we already did but were unsuccesful
            if string in self.already_solved_false:
                solved = False
            else:
                solved = solve_sudoku.solve(string)
            if solved is False:
                # Saves this sudoku as false so we don't have to try to solve it every frame
                self.already_solved_false.append(string)

                if self.already_solved.keys():

                    guesses = process.extract(string, self.already_solved.keys())

                    if guesses:

                        # Prioritizes length, then similarity to the guess
                        if approximate is False:
                            best = max(guesses, key=lambda x: (x[1], len(self.already_solved_numbers[x[0]])))[0]
                            return self.already_solved[best], self.already_solved_numbers[best]
                        else:
                            sorty = sorted(guesses, key=lambda x: (len(self.already_solved_numbers[x[0]]), x[1]), reverse=True)
                            for item in sorty:
                                if item[1] > approximate:
                                    # Sort them by length and then get the one with biggest length that has addecuate ratio?
                                    return self.already_solved[item[0]], self.already_solved_numbers[item[0]]
                            else:
                                best = max(guesses, key=lambda x: (x[1], len(self.already_solved_numbers[x[0]])))[0]
                                return self.already_solved[best], self.already_solved_numbers[best]

            # Only saves correct solutions
            if solved is not False:
                # also save the numbers that already exist in the array
                # (so we don't write over them if we can't see them)
                self.already_solved_numbers[string] = self.get_existing_numbers()
                self.already_solved[string] = solved

                return solved, self.already_solved_numbers[string]

        return False, False

    def solve(self, img_cropped_sudoku, approximate=False,required_num_in_sol=""):
        solution, existing_numbers = self.solve_by_approximate(approximate)
        self.write_solution(img_cropped_sudoku, solution, ignore=existing_numbers ,required_num_in_sol =required_num_in_sol)
        #string = self.as_string()
        return solution, existing_numbers


class Grid:
    def __init__(self):
        # physical_position  = pixel pos of center in grid , pos to to write
        self.accuracy=0
        self.image = None
        self.number = 0
        self.prev_guesses = RingBuffer(capacity=5, dtype=(float, (10)))

        self.fontsize = 0
        self.grid_position = (0, 0)
        self.grid_physical_position = (0, 0)

        self.n = 0

        # Guesses the number every self.maxtimer frames (10),
        # to reduce noise and not overuse resources 
        self.maxtimer = 10
        self.timer = self.maxtimer-1

    def update(self, image, grid_position, physical_position):
        self.image = image
        self.grid_position = grid_position

        top, right, bottom, left = physical_position
        average_dimension = (bottom-top + right-left)/2

        # NOTE edit this for better fontsize, positioning of the number
        self.fontsize = average_dimension/40
        self.n = average_dimension/4

        # NOTE edit this for better positioning of the number
        self.physical_position = (physical_position[3]+1+int(self.fontsize*self.n),
                                  physical_position[2]-int(self.fontsize*self.n))


    # def guess_number(self, confidence_threshold=0):
    #     # Saves a buffer of guesses 
    #     # Guesses every self.maxtimer frames( i.e predicts once in 10 frames)
    #     self.timer += 1
    #     if self.timer >= self.maxtimer:
    #         self.timer = 0

    #         if self.image is None:
    #             self.prev_guesses.appendleft(np.array([1,0,0,0,0,0,0,0,0,0]))
    #         else:
    #             neuron = NeuralModel.instance()
    #             prediction ,number,accuracy = neuron.guess(self.image)
    #             self.accuracy=accuracy
    #             self.prev_guesses.appendleft(np.array(prediction))

    #     m = np.mean(self.prev_guesses, axis=0)
    #     number = np.argmax(m, axis=0)

    #     self.number = number
        
    #     if m[number] > confidence_threshold:
    #         self.number = number
            
    #     return self.number
    def guess_number(self, kind=2, confidence_threshold=0):
        guy = NeuralModel.instance()
        prediction ,number,accuracy = guy.guess(self.image)
        self.accuracy=accuracy
        self.number = number
                        
        return self.number




    def write(self, sudoku_image, text):
        'Writes the given number into the position of the grid obtained '
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(sudoku_image, text, tuple(self.physical_position),
                    font, self.fontsize,(220, 90, 20), 1, cv2.LINE_AA) #130,60,0   #220, 108, 50
