# from ast import Not
import pandas as pd
import numpy as np

TRAINING_BASELINE = 0.8 # 80 Percent of images must be passed with a minimum of at least 5 images labelled (4/5 must be labelled perfectly the first time). (Total 21, train=10, test = 11)

class LabelScore():
    def __init__(self, ):
        # self.scores = pd.DataFrame(columns=("Filename", "Distance", "Score_Limit", "Nearest", "Passed", "Order_Index", "Training", "Testing"))
        self.scores = pd.DataFrame(columns=["Filename", 
                                            "Index", 
                                            "Measurment (cm)", 
                                            "Passed", 
                                            "Current Score", 
                                            "Training"])
        self.recorded_length = 0

        self.training_score = 0
        self.number_trained = 0
        self.unique_trained = 0
        self.correct_train = 0
        self.training_pass_limit = 5
        self.training_files_passed = set()
        self.train_files_tried = set()

        # Testing Results
        self.testing_score = 0
        self.number_tested = 0
        self.unique_tested = 0
        self.total_test_files = 0
        self.correct_tested = 0
        self.testing_files_passed = set()

        self.testing_file_number = None
        self.pass_threshold = None

        self.training_phase = True # False is Testing Phase

        self.testing_queue = []


    def passed_training(self):
        """
        Determines when you are complete the training set. Requires 80% accuracy on at least 5 tests.
        """
        passed = self.training_score >= 0.8 and self.number_trained >= self.training_pass_limit
        return passed

    def queue_score(self, polygon):
        # This function is designed to queue testing polygons in a specific 
        # order so that results are not needed to be calculated right away
        self.testing_queue.append(polygon)

    def add_polyscore(self, filename, measurment, passed, Training=True):

        # Increase the index of recorded
        # This will increase as it will record every time next is pressed. 
        # This will maintain progress through the entire experiment
        self.recorded_length += 1

        if Training:
            self.train_files_tried.add(filename)
            if passed:
                self.training_files_passed.add(filename)
        

            # # Length of trained files recorded
            self.number_trained = len(self.train_files_tried)
            print(self.number_trained)

            # Sum of passed files that are true
            self.correct_train = len(self.training_files_passed)

            self.training_score = self.correct_train/self.number_trained
            current_score = self.training_score

        # This does not need to be as complex because we will not repeat any files.
        else: # Testing
            self.number_tested += 1
            if passed:
                self.correct_tested += 1
            self.testing_score = self.correct_tested/self.number_tested
            current_score = self.testing_score
        
        data = [filename, self.recorded_length, measurment, passed, current_score,  Training]

        # Append data to dataframe
        self.scores.loc[0] = data




    def add_score(self, filename, training_poly, testing_poly, distance, score_limit, Training=True):
        '''
        Records scores for training and testing. 

        When a user is training and testing they will produce a score.

        Filename indicates the file that was labelled.

        distance is the euclidean distance from the measure

        Score limit is the limit to decide weather score is valid or invalid
        
        Nearest is the closest training line to any of the testing set.

        Passed is True if distance is below score limit.

        Training is weather this label was used in training or testing (Training = True, Testing = False)
        '''

            
        self.recorded_length += 1
        closest = None
        passed = bool(distance < score_limit)


        if Training:
            self.number_trained += 1
            if passed:
                self.correct_train += 1
            self.training_score = self.correct_train/self.number_trained

        else:
            self.number_tested += 1
            if passed:
                self.correct_tested += 1
            self.testing_score = self.correct_tested/self.number_tested

        # self.scores.loc[len(self.scores.index)] = [filename, training_line, testing_line, distance, score_limit, passed, self.recorded_length, Training, not(Training),]
        # self.scores.loc[len(self.scores.index)] = [filename, training_line,  passed, self.recorded_length, Training, not(Training)]


    def export_scores(self, filename):
        self.scores.to_csv(filename, index=True)
        
    

if __name__ == "__main__":
    print("yes")