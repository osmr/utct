import logging
import os

from keras.callbacks import ModelCheckpoint
from keras import backend as K

from utct.common.trainer_template import TrainerTemplate


class Trainer(TrainerTemplate):

    def _hyper_train_target_sub(self, **kwargs):

        if self.saver.log_filename:
            fh = logging.FileHandler(self.saver.log_filename)
            self.logger.addHandler(fh)

        self.logger.info("Training with parameters: {}".format(kwargs))

        X_train, Y_train, X_val, Y_val = self.data_source(**kwargs)

        K.set_image_dim_ordering('tf')

        model = self.model(input_shape=(X_train.shape[1], X_train.shape[2], 1), **kwargs)

        model.compile(loss='categorical_crossentropy', 
                      optimizer=self.optimizer(**kwargs), 
                      metrics=['accuracy'])

        callback = ModelCheckpoint(
            filepath = "{}.h5".format(
                os.path.join(
                    self.saver.last_checkpoints_dirname,
                    self.saver.model_filename_prefix)),
            monitor='val_acc',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=1)

        model.fit(x=X_train,
                  y=Y_train,
                  batch_size=self.data_source.batch_size,
                  nb_epoch=self.num_epoch,
                  verbose=1,
                  callbacks=[callback],
                  validation_data=(X_val, Y_val))

        if self.saver.log_filename:
            self.logger.removeHandler(fh)
            fh.close()

        best_value = 0.0

        return best_value