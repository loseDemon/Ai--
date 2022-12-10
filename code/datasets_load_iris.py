from sklearn import datasets
import pandas as pd
from pandas import DataFrame

x_data = datasets.load_iris().data  # .data会返回iris数据集所有的输入特征
y_data = datasets.load_iris().target  # .target会返回数据集的所有标签

print("x_data from datasets:\n", x_data)
print("y_data from datasets:\n", y_data)

x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])  # 为数据增加中文标签
pd.set_option('display.unicode.east_asian_width', True)  # 设置列名对齐
print("X_data add index \n", x_data)

x_data['类别'] = y_data  # 新加一列，列标签为'类别'，数据为y_data

print("x_data add a column: \n", x_data)
