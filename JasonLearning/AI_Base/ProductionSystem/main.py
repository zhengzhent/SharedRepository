# 通过传入的列表寻找结果
def find_data(process_data_list):
    # 依次进行循环查找并对过程排序
    for epoch, data_process in enumerate(data_process_list):
        # 用于判断此过程是否成立
        num = 0
        for i in process_data_list:
            if i in data_process:
                num += 1
        # 过程成立则数值相同，可以进入下一步
        if num == len(data_process):
            # 此过程中结果是否为最终结果，不是将此过程结果加入到过程中
            if data_result_list[epoch] not in result_list:
                # 弹出过程和此过程结果，因为此过程已经进行过，此结果存入需要查找的过程中
                result = data_result_list.pop(epoch)
                process = data_process_list.pop(epoch)
                # 判断结果是否已经存在过程中，存在则重新寻找，不存在则加入过程，并将其存入最终结果
                if result not in process_data_list:
                    dict_input['，'.join(process)] = result
                    end_result = find_data(process_data_list + [result])
                    if end_result == 1:
                        return 1
                    else:
                        return 0
                # 存在则直接寻找
                else:
                    end_result = find_data(process_data_list)
                    if end_result == 1:
                        return 1
                    else:
                        return 0
            # 找到最终结果，取出结果后返回
            else:
                process = data_process_list.pop(epoch)
                dict_input['，'.join(process)] = data_result_list[epoch]
                return 1


if __name__ == '__main__':
    # 用于储存中间过程
    data_process_list = []
    # 用于存储过程对应的结果
    data_result_list = []
    # 存储用于查询的数据
    list_data = []
    # 用于存储输出结果
    dict_input = {}

    # 规则库
    txt = '''
英超第一，是曼城队
美职联第二，是迈阿密国际队
沙特第六，是利雅得胜利队
法甲第一，是巴黎圣日尔曼队
英超第八，是曼联队
是曼城队，17号，中场，德布劳内
是迈阿密国际队，30号，中锋，梅西
是曼城队，10号，边锋，格拉利什
是利雅得胜利队，7号，前锋，C罗
是巴黎圣日耳曼队，7号，前锋，姆巴佩
是曼联队，6号，后卫，马奎尔

'''
    # 将数据预处理
    datas = txt.split('\n')
    for data in datas:
        data = data.split('，')
        data_process_list.append(data[:-1])
        data_result_list.append(data[-1].replace('\n', ''))
    # 最终结果列表
    result_list = ['德布劳内', '梅西', '格拉利什', 'C罗', '姆巴佩', '马奎尔']
    # 数据库对应的过程
    database = {'1': '英超第一', '2': '美职联第二', '3': '沙特第六', '4': '法甲第一', '5': '英超第八', '6': '是曼城队', '7': '是利雅得胜利队',
                '8': '是巴黎圣日尔曼队', '9': '是曼联队', '10': '7号', '11': '30号', '12': '10号', '13': '6号', '14':'17号','15': '中场',
                '16': '中锋', '17': '边锋', '18': '前锋', '19': '后卫', '20': '德布劳内', '21': '梅西', '22': '格拉利什', '23': 'C罗',
                '24': '姆巴佩', '25': '马奎尔'}
    # 循环进行输入，直到碰见0后退出
    print("请选择已知事实：\n选择球队排名：1: 英超第一, 2: 美职联第二, 3: 沙特第六, 4: 法甲第一, 5: 英超第八, \n选择球队：6: 是曼城队, 7: 是利雅得胜利队,8: 是巴黎圣日尔曼队, 9: 是曼联队,\n 选择球星号码：10: 7号, 11: 30号, 12: 10号, 13: 6号,14:17号\n选择球星位置： 15: 中场,16: 中锋, 17: 边锋, 18: 前锋, 19: 后卫")
    while 1:
        term = input("")
        if term == '0':
            break
        if database[term] not in list_data:
            list_data.append(database[term])
    # 打印前提条件
    print('前提条件为：')
    print(' '.join(list_data) + '\n')
    # 进行递归查找，直到找到最终结果,返回1则找到最终结果
    end_result=find_data(list_data)
    if end_result == 1:
        print('推理过程如下：')
        # 将结果进行打印
        for i in dict_input.keys():
            print(f"{i}->{dict_input[i]}")
            # 得到最终结果即输出所识别球星
            if dict_input[i] in result_list:
                print(f'所识别的球星为{dict_input[i]}')
    else:
        # 将结果进行打印
        for i in dict_input.keys():
            print(f"{i}->{dict_input[i]}")
