import json

data = json.load(open('ner.json'))

processed_data = []

for d in data:
    text = d['text']
    text_id = d['text_id']
    anns = d['ann']
    
    label = ['O'] * len(text)
    for ann in anns:
        start = ann['start']
        end = ann['end']
        label[start] = "B-" + ann['label']
        label[start+1:end] = ['I-' + ann['label']]  * (end - start - 1)    
    processed_data.append({'text': text, 'label': label, 'text_id': text_id})

json.dump(processed_data, open('ner_processed.json', 'w', encoding='utf-8'),ensure_ascii=False)