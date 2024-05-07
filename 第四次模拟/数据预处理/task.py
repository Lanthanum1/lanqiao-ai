# conda install openpyxl lxml
import pandas as pd


def document_merge() -> dict:
    data = {}
    January, February, May = {}, {}, {}
    excel_file = pd.read_excel("./2022_january.xlsx")
    # for i in range(31):
    #     January["1-"+str(excel_file.iloc[i+4,2].day)] ={excel_file.iloc[3,3]:excel_file.iloc[i+4,3],excel_file.iloc[3,4]:excel_file.iloc[i+4,4],excel_file.iloc[3,5]:excel_file.iloc[i+4,5],excel_file.iloc[3,6]:excel_file.iloc[i+4,6],excel_file.iloc[3,7]:excel_file.iloc[i+4,7],excel_file.iloc[3,8]:excel_file.iloc[i+4,8]}
    
    January = {
    f"1-{excel_file.iloc[i+4, 2].day}": {
        excel_file.iloc[3, j]: excel_file.iloc[i+4, j]
        for j in range(3, 9)
    }
    for i in range(31)
}
    

    json_file = pd.read_json("./2022_february.json")
    for i in range(1, 29):
        mid_data = "2-" + str(i)
        February[mid_data] = json_file["february"][mid_data]

    html_file = pd.read_html("./2022_may.html")[0]
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
    May = {level_data[0]: {first[j]: level_data[j] for j in range(1, 7)} 
       for i in range(1, 32) 
       for level_data in [html_file.columns.get_level_values(i).values]}

    data["january"] = January
    data["february"] = February
    data["may"] = May
    return data


print(document_merge())
