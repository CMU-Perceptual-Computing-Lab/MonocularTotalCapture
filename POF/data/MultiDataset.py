import tensorflow as tf


class MultiDataset(object):
    #  A class to combine multi dataset input
    def __init__(self, db_list):
        assert type(db_list) == list and len(db_list) >= 1
        self.db_list = db_list

    def get(self, name_wanted):
        data_list = []
        for i, db in enumerate(self.db_list):
            data = db.get()
            data_list.append(data)

        ret_data = {}
        for name in name_wanted:
            ret_data[name] = tf.concat([d[name] for d in data_list], axis=0)

        return ret_data


def combineMultiDataset(data_list, name_wanted):
    # data_list is a list of data_dict
    ret_data = {}
    for name in name_wanted:
        ret_data[name] = tf.concat([d[name] for d in data_list], axis=0)

    return ret_data


if __name__ == '__main__':
    pass
