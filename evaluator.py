# !pip install levenshtein
# #https://maxbachmann.github.io/Levenshtein/levenshtein.html#distance
# !pip install g2p_en
# #https://github.com/Kyubyong/g2p
import os
import pandas as pd
import Levenshtein as lev
from g2p_en import G2p

class Evaluator():
    def __init__(self, pred_path, actual_path):
        self.pred_path = pred_path
        self.actual_path = actual_path
        super().__init__()

    @staticmethod
    def get_target_arpabet():
        '''Return ARPABET of the given paragraph'''
        TARGET_TEXT = '''Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.'''
        ## Grapheme To Phoneme Conversion
        g2p = G2p()
        target_p = ''.join(g2p(TARGET_TEXT))
        print('Target arpabet is {}'.format(target_p))
        return target_p

    @staticmethod
    def calculate_distance(pred_p, target_p, weights=(1, 1, 1), score_cutoff=None):
        '''
        Levenshtein distance is from S1 to S2, here is from pred_p to target_p
        weights for the three operations in the form (insertion, deletion, substitution)
        If the distance is bigger than score_cutoff, score_cutoff + 1 is returned instead
        '''
        dist = lev.distance(pred_p, target_p, weights=weights, score_cutoff=score_cutoff)
        return dist

    @staticmethod
    def read_txt(input_file):
        with open(input_file, "r") as file:
            text = file.read()
        return text

    def process_files(self):
        res = []
        pred_file_list = os.listdir(self.pred_path)
        actual_file_list = os.listdir(self.actual_path)
        for file_name in pred_file_list:
            target_p = self.get_target_arpabet()
            pred_p = self.read_txt(os.path.join(self.pred_path, file_name))
            dist = self.calculate_distance(pred_p, target_p)
            dist2 = self.calculate_distance(pred_p, target_p, weights=(1, 1, 2))
            print(f'Processed file {file_name}, distance {dist} and distance 2 is {dist2}')
            one_row = {'file': file_name, 'pdist': dist, 'pdist2': dist2}
            actual_p = self.read_txt(os.path.join(self.actual_path, file_name))
            dist_a = self.calculate_distance(actual_p, target_p)
            dist2_a = self.calculate_distance(actual_p, target_p, weights=(1, 1, 2))
            one_row.update({'adist': dist_a, 'adist2': dist2_a})
            res.append(one_row)

        return pd.DataFrame(res)

    @staticmethod
    def save(df, file_path='res.csv'):
        df.to_csv(file_path, index=False)
        return


# if __name__ == '__main__':
#     pass
    # file_list = ["agni 1.wav"]
    # df = process_files(file_list)
    # df.to_csv('res.csv', index=False)