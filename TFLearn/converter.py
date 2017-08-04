import h5py
import tensorflow as tf
import tflearn

class Converter(object):

    @staticmethod
    def export_to_h5(model,
                     optimizer,
                     checkpoint_path,
                     dst_filepath,
                     **kwargs):

        config = tflearn.init_graph()
        config.gpu_options.allow_growth = True

        net = model(
            optimizer(),
            data_augmentation=None,
            **kwargs)
        model = tflearn.DNN(network=net,
                            tensorboard_verbose=0,
                            checkpoint_path=checkpoint_path,
                            max_checkpoints=2)
        model.load(model_file=checkpoint_path, weights_only=True)

        h5f = h5py.File(dst_filepath, 'w')
        var_list = tf.trainable_variables()
        for var in var_list:
            name = var.name.split(':')[0]
            data = model.get_weights(var)
            if name.endswith('/W'):
                if len(data.shape) == 4:
                    #data = data.transpose((3, 2, 0, 1))
                    data = data.transpose((3, 2, 1, 0))
                elif len(data.shape) == 2:
                    data = data.transpose((1, 0))
            h5f.create_dataset(
                name=name,
                data=data)
        h5f.close()

    @staticmethod
    def import_from_h5(model,
                       optimizer,
                       src_filepath,
                       checkpoint_path,
                       **kwargs):

        config = tflearn.init_graph()
        config.gpu_options.allow_growth = True

        net = model(
            optimizer(),
            data_augmentation=None,
            **kwargs)
        model = tflearn.DNN(network=net,
                            tensorboard_verbose=0,
                            checkpoint_path=checkpoint_path,
                            max_checkpoints=2)

        h5f = h5py.File(src_filepath, 'r')
        for k, v in h5f.items():
            if isinstance(v, h5py.Group):
                for k2, v2 in v.items():
                    name = k
                    # vars = tflearn.get_layer_variables_by_name(name)
                    data = v2[:]
                    if k2 == 'b':
                        name += "/b:0"
                        # tflearn.set_value(a[1], data, model.session)
                        # model.set_weights(vars[1], data)
                    if k2 == 'W':
                        name += "/W:0"
                        if len(data.shape) == 4:
                            #data = data.transpose((2, 3, 1, 0))
                            #print data.shape
                            data = data.transpose((3, 2, 1, 0))
                            #print data.shape
                        elif len(data.shape) == 2:
                            data = data.transpose((1, 0))
                            # tflearn.set_value(a[0], data, model.session)
                            # model.set_weights(vars[0], data)
                    #tensor = tf.get_default_graph().get_tensor_by_name(name)
                    tensor = model.session.graph.get_tensor_by_name(name)
                    model.set_weights(tensor, data)
            else:
                raise NotImplementedError()
        h5f.close()

        model.save(model_file=checkpoint_path)
