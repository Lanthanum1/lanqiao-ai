#task-start
import random
import pandas as pd
from torch.utils.data import Dataset


class MakeDataset(Dataset):

    def __init__(self):
        self.data = pd.read_csv('data.csv')[['text_id', 'text']].values
        # 这个.values就是把对象转换为array,不能缺少，但是至于[['text_id', 'text']]因为原本csv中只有这两列，实际上是可以缺省的
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
        match_loc=[]
        
        # 遍历每个地点，如果text中包含该地点，则将该地点加入match_loc
        for loc in self.locs:
            if loc in text:
                match_loc.append(loc)

        for loc in match_loc:
            replacement_loc = random.choice(self.locs)
            text=text.replace(loc,replacement_loc)
            aug_info['locs'].append({'original':loc,'replacement':replacement_loc})

        match_per=[]
        for per in self.pers:
            if per in text:
                match_per.append(per)
                
        for per in match_per:
            replacement_per =random.choice(self.pers)
            text=text.replace(per,replacement_per)
            aug_info['pers'].append({'original':per,'replacement':replacement_per})

        return text, aug_info




def main():
    dataset = MakeDataset()
    for data in dataset:
        print(data)

if __name__ == '__main__':
    main()
#task-end