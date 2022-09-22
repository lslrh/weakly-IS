import json

def get_json_data():

    dicts = {} # 用来存储数据
    with open('./voc_2012_train.json','r',encoding='utf8') as f:
        json_data = json.load(f)
        annotations = json_data["annotations"]
        for annotation in annotations:
            annotation["category_id"] -= 1
        
        dicts = json_data # 将修改后的内容保存在dict中      
    return dicts


def write_json_data(dict):#写入json文件
    with open('./voc_v2','w') as r:
        json.dump(dict,r)
 
file = get_json_data()
write_json_data(file)

