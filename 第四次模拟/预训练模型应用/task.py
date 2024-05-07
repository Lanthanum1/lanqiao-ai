import json
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


in_t = "inception_train.json"
re_t = "resnet_train.json"
xc_t = "xception_train.json"

in_t = json.load(open(in_t))
re_t = json.load(open(re_t))
xc_t = json.load(open(xc_t))

# 遍历re_t字典，将in_t和xc_t中相同键的"feature"字段合并到一起
for r in re_t:
    re_t[r]["feature"] = re_t[r]["feature"] + in_t[r]["feature"] + xc_t[r]["feature"]

x = [u["feature"] for u in re_t.values()] # 存储所有样本的特征
y = [u["label"] for u in re_t.values()] # 存储所有样本的标签



x = np.array(x)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
# random_state参数是一个可选的设置，用于控制数据分割时随机数生成器的种子（seed）。

# 当random_state被设定为一个固定的值，比如42，这意味着每次执行train_test_split时，都会得到完全相同的训练集和测试集划分。

# 如果省略random_state或者设置为None，那么在每次调用函数时，由于使用了不同的随机种子，数据的划分将会不同，导致结果具有随机性。

clf = CatBoostClassifier(
    iterations=10, # 5也能通过测试
    learning_rate=0.1,
    depth=6,
    loss_function="Logloss",
    logging_level="Silent",
    # logging_level="Verbose",
)
clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)

with open("test.json", "r") as f:
    f = json.load(f)
    with open("result.csv", "w") as g:
        g.write("id,label\n")
        for k, v in f.items():
            feature = v["feature"]
            feature = np.array(feature).reshape(1, -1) # -1告诉NumPy自动推算列数。
            label = clf.predict(feature)
            g.write(k + "," + str(label[0]) + "\n")
