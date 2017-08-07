import h5py
import logging
import mxnet as mx
# import numpy as np


class Converter(object):
    """
    Class, which provides saving (and loading) models' weights into universal format.
    """

    @staticmethod
    def export_to_h5(checkpoint_path,
                     checkpoint_epoch,
                     dst_filepath):
        """
        Saving models' weights into universal format.

        Parameters:
        ----------
        checkpoint_path : str
            path to checkpoint file with the prefix
        checkpoint_epoch : int
            number of epoch for the checkpoint file
        dst_filepath : str
            path to output h5 file
        """

        sym, arg_params, aux_params = mx.model.load_checkpoint(
            prefix=checkpoint_path,
            epoch=checkpoint_epoch)

        h5f = h5py.File(dst_filepath, 'w')
        for k, v in arg_params.items():
            name = k
            if name.endswith('_bias'):
                name = name[:-5] + "/b"
            if name.endswith('_weight'):
                name = name[:-7] + "/W"
            data = v.asnumpy()

            # #data = np.arange(0, np.prod(np.array(data.shape)))
            # data = np.random.random_sample(np.prod(np.array(data.shape)))
            # np.random.shuffle(data)
            # data = data.reshape(v.asnumpy().shape)

            h5f.create_dataset(name=name, data=data)
        h5f.close()

    @staticmethod
    def import_from_h5(model,
                       data_source,
                       src_filepath,
                       checkpoint_path,
                       checkpoint_epoch,
                       ctx,
                       **kwargs):
        """
        Loading models' weights from universal format.

        Parameters:
        ----------
        model : object
            instance of Model class with graph of CNN
        data_source : object
            instance of DataSource class with training/validation iterators
        src_filepath : str
            path to input h5 file
        checkpoint_path : str
            path to checkpoint file with the prefix
        checkpoint_epoch : int
            number of epoch for the checkpoint file
        ctx : object
            instance of MXNet context
        """

        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        mod = mx.mod.Module(
            symbol=model(**kwargs),
            logger=logger,
            context=ctx)

        arg_params = {}
        h5f = h5py.File(src_filepath, 'r')
        for k, v in h5f.items():
            if isinstance(v, h5py.Group):
                for k2, v2 in v.items():
                    name = k
                    if k2 == 'b':
                        name += "_bias"
                    if k2 == 'W':
                        name += "_weight"
                    arg_params[name] = mx.nd.array(v2)
            else:
                arg_params[k] = mx.nd.array(v)
        h5f.close()

        mod._arg_params = arg_params
        mod._aux_params = {}
        mod.params_initialized = True

        train_iter, val_iter = data_source(**kwargs)
        mod.bind(
            data_shapes=train_iter.provide_data,
            label_shapes=train_iter.provide_label,
            for_training=False)

        mod.save_checkpoint(
            prefix=checkpoint_path,
            epoch=checkpoint_epoch)
