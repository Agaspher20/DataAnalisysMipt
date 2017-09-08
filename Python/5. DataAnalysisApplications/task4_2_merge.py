""" Merge sort module """

def merge(left_list, left_length, right_list, right_length, key, descending):
    """ applies ordered merging of two lists based on key """
    left_index = 0
    right_index = 0
    result = []
    if descending:
        less_than = (lambda right, left: right > left)
    else:
        less_than = (lambda right, left: right < left)
    while left_index < left_length and right_index < right_length:
        left = left_list[left_index]
        right = right_list[right_index]
        if less_than(key(right), key(left)):
            result.append(right)
            right_index += 1
        else:
            result.append(left)
            left_index += 1
    if left_index == left_length:
        return result + right_list[right_index:right_length]
    else:
        return result + left_list[left_index:left_length]

def merge_sort(data, key, descending=False):
    """ Applies merge sort in data based on key """
    list_length = len(data)
    if list_length > 1:
        left_length = int(list_length/2)
        right_length = list_length-left_length
        left_list = merge_sort(data[0:left_length], key, descending)
        right_list = merge_sort(data[left_length:list_length], key, descending)

        return merge(left_list, left_length, right_list, right_length, key, descending)
    else:
        return data
