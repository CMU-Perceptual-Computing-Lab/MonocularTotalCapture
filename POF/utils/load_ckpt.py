import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def load_weights_from_snapshot(session, checkpoint_path, discard_list=None, rename_dict=None):
        """ Loads weights from a snapshot except the ones indicated with discard_list. Others are possibly renamed. """
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()

        # Remove everything from the discard list
        if discard_list is not None:
            num_disc = 0
            var_to_shape_map_new = dict()
            for k, v in var_to_shape_map.items():
                good = True
                for dis_str in discard_list:
                    if dis_str in k:
                        good = False

                if good:
                    var_to_shape_map_new[k] = v
                else:
                    num_disc += 1
            var_to_shape_map = dict(var_to_shape_map_new)
            print('Discarded %d items' % num_disc)

        # rename everything according to rename_dict
        num_rename = 0
        var_to_shape_map_new = dict()
        for name in var_to_shape_map.keys():
            new_name = name
            if rename_dict is not None:
                for rename_str in rename_dict.keys():
                    if rename_str in name:
                        new_name = new_name.replace(rename_str, rename_dict[rename_str], 1)  # my modification: replace no more than once
                        num_rename += 1
            var_to_shape_map_new[new_name] = reader.get_tensor(name)
        var_to_shape_map = dict(var_to_shape_map_new)

        init_op, init_feed = tf.contrib.framework.assign_from_values(var_to_shape_map)
        session.run(init_op, init_feed)
        print('Initialized %d variables from %s.' % (len(var_to_shape_map), checkpoint_path))


def load_weights_to_dict(checkpoint_path, discard_list=None, rename_dict=None):
    """ Loads weights from a snapshot except the ones indicated with discard_list. Others are possibly renamed. """
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Remove everything from the discard list
    if discard_list is not None:
        num_disc = 0
        var_to_shape_map_new = dict()
        for k, v in var_to_shape_map.items():
            good = True
            for dis_str in discard_list:
                if dis_str in k:
                    good = False

            if good:
                var_to_shape_map_new[k] = v
            else:
                num_disc += 1
        var_to_shape_map = dict(var_to_shape_map_new)
        print('Discarded %d items' % num_disc)

    # rename everything according to rename_dict
    num_rename = 0
    var_to_shape_map_new = dict()
    for name in var_to_shape_map.keys():
        new_name = name
        if rename_dict is not None:
            for rename_str in rename_dict.keys():
                if rename_str in name:
                    new_name = new_name.replace(rename_str, rename_dict[rename_str])
                    num_rename += 1
        var_to_shape_map_new[new_name] = reader.get_tensor(name)
    var_to_shape_map = dict(var_to_shape_map_new)

    return var_to_shape_map
