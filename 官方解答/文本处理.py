#task-start
import random
import pandas as pd
from torch.utils.data import Dataset


class MakeDataset(Dataset):

    def __init__(self):
        self.data = pd.read_csv('data.csv')[['text_id', 'text']].values
        self.locs = open('loc.txt', 'r').read().split('\n')
        self.pers = open('per.txt', 'r').read().split('\n')

    def __getitem__(self, item):
        text_id, text = self.data[item]
        text, aug_info = self.augment(text)
        return text_id, text, aug_info

    def __len__(self):
        return len(self.data)

    def augment(self, text):
        aug_info = {'locs': [], 'pers': []}

        # TODO
        matach_loc = []
        for loc in self.locs:
            if loc in text:
                matach_loc.append(loc)
        for loc in matach_loc:
            rep_loc = random.choice(self.locs)
            text = text.replace(loc, rep_loc)
            aug_info['locs'].append({"original":loc, "replacement":rep_loc})
            
        matach_per = []
        for per in self.pers:
            if per in text:
                matach_per.append(per)
        for per in matach_per:
            rep_per = random.choice(self.pers)
            text = text.replace(per, rep_per)
            aug_info['pers'].append({"original":per, "replacement":rep_per})
            
        # test
        

        return text, aug_info


def main():
    dataset = MakeDataset()
    for data in dataset:
        print(data)

if __name__ == '__main__':
    main()
#task-end