# task-start
from flask import Flask, jsonify, request
import torch

app = Flask(__name__)
model = torch.jit.load("ner.pt")
model.eval()
index2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC"}


def process(inputs):

    # TODO
    results = []
    outputs = model(torch.tensor(inputs)).detach().numpy()
    # 这个outputs是模型的输出，是一个二维数组，每个元素是一个数组，数组的每个元素是一个数字，代表标签的索引,如[[0 3 4 1 2...], [0 3 4 0 0...],...]
    for sentence in outputs:
        result = []
        i = 0
        while i < len(sentence):
            tag = index2label[sentence[i]]
            if tag.startswith("B-"):
                # 因为单字实体 'O' 和 以 I- 起始的实体不需要标注，所以判断startswith("B-")开始标注
                entity_type = tag[2:]  # 取出"B-"后面的实体类型
                start = i
                i += 1
                while (
                    i < len(sentence) and index2label[sentence[i]] == f"I-{entity_type}"
                ):
                    # 因为"B-"本身可以作为实体的一部分，所以这里判断后面是不是"I-",并且要求类型一致
                    i += 1
                if i - start > 1:
                    # 因为题目要求单字实体'B-LOC'作为非实体序列处理，不需要标注，所以这里加入判断大于一
                    result.append({"start": start, "end": i - 1, "label": entity_type})
            else:
                i += 1
        results.append(result)
    return results


@app.route("/ner", methods=["POST"])
def ner():

    data = request.get_json()
    inputs = data["inputs"]
    outputs = process(inputs)
    return jsonify(outputs)


if __name__ == "__main__":
    app.run(debug=True)
# task-end
