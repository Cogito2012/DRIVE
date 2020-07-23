import os
import numpy as np

if __name__ == "__main__":
    data_path = './DADA-2000'
    output_txt = os.path.join(data_path, 'stat.txt')

    # train set
    train_path = os.path.join(data_path, 'training', 'rgb')
    stat_train = np.zeros((54), dtype=np.int32)
    for clsName in sorted(os.listdir(train_path)):
        clsID = int(clsName) - 1
        stat_train[clsID] = len(os.listdir(os.path.join(train_path, clsName)))

    # val set
    val_path = os.path.join(data_path, 'validation', 'rgb')
    stat_val = np.zeros((54), dtype=np.int32)
    for clsName in sorted(os.listdir(val_path)):
        clsID = int(clsName) - 1
        stat_val[clsID] = len(os.listdir(os.path.join(val_path, clsName)))

    # test set
    test_path = os.path.join(data_path, 'testing', 'rgb')
    stat_test = np.zeros((54), dtype=np.int32)
    for clsName in sorted(os.listdir(test_path)):
        clsID = int(clsName) - 1
        stat_test[clsID] = len(os.listdir(os.path.join(test_path, clsName)))
    
    from terminaltables import AsciiTable
    display_data = [["cls ID"], ["# train"], ["# val"], ["# test"]]
    num_train, num_val, num_test = 0, 0, 0
    for clsID, (n_train, n_val, n_test) in enumerate(zip(stat_train, stat_val, stat_test)):
        if n_train >= 10 and n_val >= 2 and n_test >= 2:
            display_data[0].append(str(clsID + 1))
            display_data[1].append(str(n_train))
            num_train += n_train
            display_data[2].append(str(n_val))
            num_val += n_val
            display_data[3].append(str(n_test))
            num_test += n_test
    display_data[0].append("# total")
    display_data[1].append(str(num_train))
    display_data[2].append(str(num_val))
    display_data[3].append(str(num_test))
    display_title = "Number of videos for each category. Thresholds: train(=10), val(=2), test(=2)"
    table = AsciiTable(display_data, display_title)
    # table.inner_footing_row_border = True
    print(table.table)

    with open(output_txt, 'w') as f:
        f.writelines(table.table)
    print("results saved in %s. "%(output_txt))