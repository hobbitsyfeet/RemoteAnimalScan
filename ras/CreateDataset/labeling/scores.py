# from ast import Not
import pandas as pd

class LabelScore():
    def __init__(self):
        self.scores = pd.DataFrame(columns=("Filename", "Distance", "Score_Limit", "Nearest", "Passed", "Order_Index", "Training", "Testing"))
        self.recorded_length = 0
    
    def add_score(self, filename, training_line, testing_line, distance, score_limit, Training=True):
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
        self.scores.loc[len(self.scores.index)] = [filename, training_line, testing_line, distance, score_limit, passed, self.recorded_length, Training, not(Training),]


    def export_scores(self, filename):
        self.scores.to_csv(filename, index=True)
        
    

if __name__ == "__main__":
    print("yes")