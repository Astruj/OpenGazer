
def reorderFlow(Dict):
    values_list=list(Dict.values())
    sorted_texts=[]
    def takeFirstSecond(elem):
        return elem[0][1]

    values_list.sort(key=takeFirstSecond)

    def takeSecond(elem):
        return elem[1]

    values_list.sort(key=takeSecond)

    for i in range(len(Dict)):
        for key, value in Dict.items():
            if values_list[i] == value:
                # print(key)
                sorted_texts.append(key)
    sorted_texts = '\n'.join(sorted_texts)
    return sorted_texts

if __name__ == "__main__":
    Dict = {'1': [[0, 1, 0, 0], 11],
            '2': [[0, 3, 0, 0], 9],
            '3': [[0, 2, 0, 0], 11],
            '4': [[0, 4, 0, 0], 8],
            '5': [[0, 3, 0, 0], 8],
            '6': [[0, 2, 0, 0], 9],
            '7': [[0, 5, 0, 0], 8]}
    st = reorderFlow(Dict)
    print(st)
