import random

def pre2region(dev_res_, dev_tag_list):
    """
    将单句分词结果转换为区间
    :param dev_res_: 预测出来的数字序列（可能不标准）
    :param dev_tag_list: 为了获得长度
    :return: “商品 和 服务[0,2,3,0,2]”将返回[(0, 2), (2, 3), (3, 5)]
    """
    region = []
    start = 0
    for i in range(len(dev_tag_list)):
        if i == len(dev_tag_list)-1:
            end = i+1
            region.append((start, end))
            break
        if dev_res_[i+1] == 0 or dev_res_[i+1] == 3:
            end = i+1
            region.append((start, end))
            start = end
        elif dev_res_[i+1] == 4 and random.randint(0, 1) == 0:
            end = i+1
            region.append((start, end))
            start = end
    return region


def ans2region(dev_tag_list):
    """
    将单句答案转换为区间
    :param dev_tag_list: 原始的数字序列（标准）
    :return: “商品 和 服务[0,2,3,0,2]”将返回[(0, 2), (2, 3), (3, 5)]
    """
    region = []
    start = 0
    for i, num in enumerate(dev_tag_list):
        if num == 'E' or num == 'S':
            end = i+1
            region.append((start, end))
            start = end
    return region

def calculate(dev_res, dev_tag_lists):
    tp, ans_total, pre_total = 0, 0, 0
    for dev_res_, dev_tag_list in zip(dev_res, dev_tag_lists):
        pre = set(pre2region(dev_res_, dev_tag_list))
        ans = set(ans2region(dev_tag_list))
        tp += len(pre&ans)
        ans_total += len(ans)
        pre_total += len(pre)

    precision = tp/pre_total
    recall = tp/ans_total

    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1