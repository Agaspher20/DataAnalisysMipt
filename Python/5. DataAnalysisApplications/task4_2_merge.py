def merge(left_list, left_length, right_list, right_length, key):
    """ applies ordered merging of two lists based on key """
    left_index = 0
    right_index = 0
    result = []
    while left_index < left_length and right_index < right_length:
        left = left_list[left_index]
        right = right_list[right_index]
        if key(right) < key(left):
            result.append(right)
            right_index += 1
        else:
            result.append(left)
            left_index += 1
    if left_index == left_length:
        return result + right_list[right_index:right_length]
    else:
        return result + left_list[left_index:left_length]

def merge_sort(data, key):
    """ Applies merge sort in data based on key """
    list_length = len(data)
    if list_length > 1:
        left_length = int(list_length/2)
        right_length = list_length-left_length
        left_list = merge_sort(data[0:left_length], key)
        right_list = merge_sort(data[left_length:list_length], key)

        return merge(left_list, left_length, right_list, right_length, key)
    else:
        return data
