# conda install openpyxl lxml
import pandas as pd
import json

def document_merge() -> dict:
    data = {}
    January, February, May = {}, {}, {}
    excel_file = pd.read_excel("/home/project/2022_january.xlsx")
    # for i in range(31):
    #     January["1-"+str(excel_file.iloc[i+4,2].day)] ={excel_file.iloc[3,3]:excel_file.iloc[i+4,3],excel_file.iloc[3,4]:excel_file.iloc[i+4,4],excel_file.iloc[3,5]:excel_file.iloc[i+4,5],excel_file.iloc[3,6]:excel_file.iloc[i+4,6],excel_file.iloc[3,7]:excel_file.iloc[i+4,7],excel_file.iloc[3,8]:excel_file.iloc[i+4,8]}
    
#     January = {
#     f"1-{excel_file.iloc[i+4, 2].day}": {
#         excel_file.iloc[3, j]: excel_file.iloc[i+4, j]
#         for j in range(3, 9)
#     }
#     for i in range(31)
# }
    January = {}

    for i in range(31):
        # 初始化每日数据的字典
        daily_data = {}
        for j in range(3, 9):
            # 从excel_file中提取数据并填充到daily_data字典中
            daily_data[excel_file.iloc[3, j]] = excel_file.iloc[i + 4, j]
        
        # 使用日期作为键，daily_data作为值，添加到January字典中
        January[f"1-{excel_file.iloc[i+4, 2].day}"] = daily_data
    

    # json_file = pd.read_json("/home/project/2022_february.json")
    json_file = json.load(open("/home/project/2022_february.json"))
    # for i in range(1, 29):
    #     mid_data = "2-" + str(i)
    #     February[mid_data] = json_file["february"][mid_data]

    html_file = pd.read_html("/home/project/2022_may.html")[0]
    # [0]是因为html中可能会有多个表格，所以要指定第几个表格
    
    first = html_file.columns.get_level_values(0).values
    # html_file.columns对象每一个元素都是表格的一个列，get_level_values(i)获取第i行的值
    
    # for i in range(1, 32):
    #     mid_data = html_file.columns.get_level_values(i).values
    #     May[mid_data[0]] = {
    #         first[1]: mid_data[1],
    #         first[2]: mid_data[2],
    #         first[3]: mid_data[3],
    #         first[4]: mid_data[4],
    #         first[5]: mid_data[5],
    #         first[6]: mid_data[6],
    #     }
    # May = {level_data[0]: {first[j]: level_data[j] for j in range(1, 7)} 
    #    for i in range(1, 32) 
    #    for level_data in [html_file.columns.get_level_values(i).values]}
    May = {}

    for i in range(1, 32):
        level_data = html_file.columns.get_level_values(i).values
        May[level_data[0]] = {}
        for j in range(1, 7):
            May[level_data[0]][first[j]] = level_data[j]

   

    data["january"] = January
    data["february"] = json_file["february"]
    data["may"] = May
    return data


# print(document_merge())
