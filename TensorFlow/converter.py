import os
import h5py
import tensorflow as tf
#from .utils import fix_saver

class Converter(object):
    """
    Class, which provides saving (and loading) models' weights into universal format.
    """

    @staticmethod
    def export_to_h5(model,
                     checkpoint_path,
                     dst_filepath,
                     **kwargs):
        """
        Saving models' weights into universal format.

        Parameters:
        ----------
        model : object
            instance of Model class with graph of CNN
        dst_filepath : str
            path to output h5 file
        """

        with tf.get_default_graph().as_default():
            session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            session = tf.Session(config=session_config)

            try:
                session.run([tf.global_variables_initializer(),
                             tf.local_variables_initializer()])
            except Exception:
                session.run(tf.initialize_all_variables())

            net, loss, metric = model(
                is_training=False,
                **kwargs)

            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), checkpoint_path))
            restorer_trainvars = tf.train.Saver(var_list=tf.trainable_variables())
            restorer_trainvars.restore(session, checkpoint_path)

            h5f = h5py.File(dst_filepath, 'w')
            var_list = tf.trainable_variables()
            for var in var_list:
                name = var.name.split(':')[0]
                data = var.eval(session)
                if name.endswith('/weights'):
                    name = name[:-8] + "/W"
                    if len(data.shape) == 4:
                        #data = data.transpose((3, 2, 0, 1))
                        data = data.transpose((3, 2, 1, 0))
                    elif len(data.shape) == 2:
                        data = data.transpose((1, 0))
                elif name.endswith('/biases'):
                    name = name[:-7] + "/b"
                h5f.create_dataset(
                    name=name,
                    data=data)
            h5f.close()

    @staticmethod
    def import_from_h5(model,
                       src_filepath,
                       checkpoint_path,
                       **kwargs):
        """
        Loading models' weights from universal format.

        Parameters:
        ----------
        model : object
            instance of Model class with graph of CNN
        src_filepath : str
            path to input h5 file
        checkpoint_path : str
            path to checkpoint file
        """

        with tf.get_default_graph().as_default():
            session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            session = tf.Session(config=session_config)

            try:
                session.run([tf.global_variables_initializer(),
                             tf.local_variables_initializer()])
            except Exception:
                session.run(tf.initialize_all_variables())

            net, loss, metric = model(
                is_training=False,
                **kwargs)

            h5f = h5py.File(src_filepath, 'r')
            for k, v in h5f.items():
                if isinstance(v, h5py.Group):
                    for k2, v2 in v.items():
                        name = k
                        # vars = tflearn.get_layer_variables_by_name(name)
                        data = v2[:]
                        if k2 == 'b':
                            name += "/biases:0"
                            # tflearn.set_value(a[1], data, model.session)
                            # model.set_weights(vars[1], data)
                        if k2 == 'W':
                            name += "/weights:0"
                            if len(data.shape) == 4:
                                # data = data.transpose((2, 3, 1, 0))
                                # print data.shape
                                data = data.transpose((3, 2, 1, 0))
                                # print data.shape
                            elif len(data.shape) == 2:
                                data = data.transpose((1, 0))
                                # tflearn.set_value(a[0], data, model.session)
                                # model.set_weights(vars[0], data)
                        # tensor = tf.get_default_graph().get_tensor_by_name(name)
                        tensor = session.graph.get_tensor_by_name(name)
                        #model.set_weights(tensor, data)
                        session.run(tf.assign(tensor, data))
                else:
                    raise NotImplementedError()
            h5f.close()

            #model.save(model_file=checkpoint_path)

            saver = tf.train.Saver()
            #obj_lists = fix_saver()
            # TF 0.12 Fix
            if not os.path.isabs(checkpoint_path):
                model_file = os.path.abspath(os.path.join(os.getcwd(), checkpoint_path))
            saver.save(session, checkpoint_path)
            #fix_saver(obj_lists)
