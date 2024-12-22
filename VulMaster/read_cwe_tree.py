import json



def line2key_values(string_):
    d = string_
    # print(d)
    # print(d.split(':')[0].strip()[1:-1], type(d.split(':')[0].strip()[1:-1]))
    # print(d.split(':')[1].strip()[1:-1], type(d.split(':')[1].strip()))
    key_ = d.split(':')[0].strip()[1:-1]
    values = d.split(':')[1].strip()[1:-1].strip().split(',')
    values = [v.strip()[1:-1] for v in values]
    new_values = []
    for v in values:
        if len(v) > 0:
            new_values.append(v)
    values = new_values

    # print('key:', key_, type(key_))
    # print('val:', values, type(values), len(values))
    return key_, values


def find_near_child_and_parent_and_peer(cur_node, parent_child_dict):
    near_child, near_parent, near_peer = [],[],[]
    if cur_node in parent_child_dict.keys():
        near_child = parent_child_dict[cur_node]
    for key_ in parent_child_dict.keys():
        if (key_ != cur_node) and (cur_node in parent_child_dict[key_]):
            near_parent = key_
            near_peer = parent_child_dict[key_]
            near_peer.remove(cur_node)


    return near_parent, near_child, near_peer


def get_near_cwe_types(cur_node):
    with open('cwe_tree.json') as f:
        data = f.readlines()
    parent_child_dict = dict()
    for i in range(len(data)):
        parent, childs = line2key_values(data[i].strip()[1:-1])
        parent_child_dict[parent] = childs
    near_parent, near_child, near_peer = find_near_child_and_parent_and_peer(cur_node, parent_child_dict)
    return near_parent, near_child, near_peer


if __name__ == "__main__":
    cur_node = 'CWE-000'
    near_parent, near_child, near_peer = get_near_cwe_types(cur_node)
    print(near_parent)
    print(near_peer)
    print(near_child)
    print(114514)


['CWE- 120', 'CWE- 125', 'CWE-466', 'CWE-680', 'CWE-786', 'CWE- 787', 'CWE-788', 'CWE-805', 'CWE-822', 'CWE-824', 'CWE-825']
