import tensorflow as tf
import numpy as np

def tf_mean_squared_error(y_est, y):
    with tf.name_scope("MSE"):
        y_est.get_shape().assert_is_compatible_with(y.get_shape())
        return tf.reduce_mean(
            tf.squared_difference(y_est, y),
            name="MSE")


def tf_mean_absolute_error(y_est, y):
    with tf.name_scope("MAE"):
        y_est.get_shape().assert_is_compatible_with(y.get_shape())
        return tf.reduce_mean(
            tf.abs(
                tf.subtract(y_est, y)),
            name="MAE")


def tf_masked_mean_squared_error(y_est, y_masked):
    with tf.name_scope("MSE"):
        y = y_masked[:, :, 0]
        y_mask = y_masked[:, :, 1]
        y_est.get_shape().assert_is_compatible_with(y.get_shape())
        return tf.div(
            tf.reduce_sum(
                tf.square(
                    tf.multiply(
                        tf.subtract(y_est, y),
                        y_mask))),
            tf.reduce_sum(y_mask),
            name="MSE")


def tf_masked_root_mean_square_error_on_mse(mse):
    with tf.name_scope("RMSE"):
        return tf.sqrt(
            mse,
            name="RMSE")


def tf_masked_root_mean_square_error(y_est, y_masked):
    with tf.name_scope("RMSE"):
        return tf.sqrt(
            tf_masked_mean_squared_error(y_est, y_masked),
            name="RMSE")


def tf_masked_scaled_root_mean_square_error(y_est, y_masked, y_std):
    with tf.name_scope("RMSE"):
        y = y_masked[:, :, 0]
        y_mask = y_masked[:, :, 1]
        y_est.get_shape().assert_is_compatible_with(y.get_shape())
        return tf.sqrt(
            tf.div(
                tf.reduce_sum(
                    tf.square(
                        tf.multiply(
                            tf.multiply(
                                tf.subtract(y_est, y),
                                y_mask),
                            y_std))),
                tf.reduce_sum(y_mask)),
            name="RMSE")


def tf_masked_scaled_root_mean_square_errors(y_est, y_masked, y_std):
    with tf.name_scope("MSE"):
        y = y_masked[:, :, 0]
        y_mask = y_masked[:, :, 1]
        y_est.get_shape().assert_is_compatible_with(y.get_shape())
        y_diff_masked = tf.multiply(
            tf.subtract(y_est, y),
            y_mask,
            name="y_diff_masked")
        y_mask_count = tf.reduce_sum(
            y_mask,
            name="y_mask_count")
        mse = tf.div(
            tf.reduce_sum(tf.square(y_diff_masked)),
            y_mask_count,
            name="MSE")
        rmse = tf.sqrt(
            tf.div(
                tf.reduce_sum(
                    tf.square(
                        tf.multiply(
                            y_diff_masked,
                            y_std))),
                y_mask_count),
            name="RMSE")
        return mse, rmse


def tf_masked_scaled_root_mean_square_errors2(y_est, y_masked, y_std):
    with tf.name_scope("MSE"):
        y = y_masked[:, :, 0]
        y_mask = y_masked[:, :, 1]
        y_est.get_shape().assert_is_compatible_with(y.get_shape())
        y_diff_masked = tf.multiply(
            tf.subtract(y_est, y),
            y_mask,
            name="y_diff_masked")
        y_mask_count = tf.reduce_sum(
            y_mask,
            name = "y_mask_count")
        mse = tf.div(
            tf.reduce_sum(tf.square(y_diff_masked)),
            y_mask_count,
            name="MSE")
        y_diff_masked_orig = tf.multiply(
            y_diff_masked,
            y_std,
            name="y_diff_masked_orig")
        rmse = tf.sqrt(
            tf.div(
                tf.reduce_sum(tf.square(y_diff_masked_orig)),
                y_mask_count),
            name="RMSE")
        mae = tf.div(
                tf.reduce_sum(tf.abs(y_diff_masked_orig)),
                y_mask_count,
                name="MAE")
        return mse, rmse, mae


def tf_neg_masked_scaled_root_mean_square_errors(y_est, y_masked, y_std, do_neg_rmdse=False):
    with tf.name_scope("MSE"):
        y = y_masked[:, :, 0]
        y_mask = y_masked[:, :, 1]
        y_est.get_shape().assert_is_compatible_with(y.get_shape())
        y_diff_masked = tf.multiply(
            tf.subtract(y_est, y),
            y_mask,
            name="y_diff_masked")
        y_mask_count = tf.reduce_sum(
            y_mask,
            name = "y_mask_count")
        mse = tf.div(
            tf.reduce_sum(tf.square(y_diff_masked)),
            y_mask_count,
            name="MSE")
        rmse = tf.sqrt(
            tf.div(
                tf.reduce_sum(
                    tf.square(
                        tf.multiply(
                            y_diff_masked,
                            y_std))),
                y_mask_count),
            name="RMSE")
        if do_neg_rmdse:
            rmse = tf.neg(rmse, name="-RMSE")
        return mse, rmse


def np_masked_scaled_root_mean_square_error(y_est, y_masked, y_std):
    y = y_masked[:, :, 0]
    y_mask = y_masked[:, :, 1]
    assert y_est.shape == y.shape
    return np.sqrt(
        np.divide(
            np.sum(
                np.square(
                    np.multiply(
                        np.multiply(
                            y_est - y,
                            y_mask),
                        y_std))),
            np.sum(y_mask)))


def np_masked_scaled_root_mean_square_error2(y_est, y_masked, y_std):
    y = y_masked[:, :, 0]
    y_mask = y_masked[:, :, 1]
    assert y_est.shape == y.shape
    return np.sqrt(
        np.divide(
            np.sum(
                np.square(
                    np.multiply(
                        np.multiply(
                            y_est - y,
                            y_mask),
                        y_std)),
                axis=1),
            np.sum(y_mask, axis=1)))


def np_masked_scaled_root_mean_square_error3(y_est, y_masked, y_std):
    y = y_masked[:, :, 0]
    y_mask = y_masked[:, :, 1]
    assert y_est.shape == y.shape
    return np.sqrt(
        np.divide(
            np.sum(
                np.square(
                    np.multiply(
                        np.multiply(
                            y_est - y,
                            y_mask),
                        y_std)),
                axis=0),
            np.sum(y_mask, axis=0)))


def tf_masked_mean_absolute_error(y_est, y_masked):
    with tf.name_scope("MAE"):
        y = y_masked[:, :, 0]
        y_mask = y_masked[:, :, 1]
        y_est.get_shape().assert_is_compatible_with(y.get_shape())
        return tf.div(
            tf.reduce_sum(
                tf.abs(
                    tf.multiply(
                        tf.subtract(y_est, y),
                        y_mask))),
            tf.reduce_sum(y_mask),
            name="MAE")


def tf_r2_score(y_est, y):
    y_est.get_shape().assert_is_compatible_with(y.get_shape())
    y_mean = tf.reduce_mean(y, 0)
    num = tf.reduce_sum(tf.squared_difference(y, y_est), 0)
    den = tf.reduce_sum(tf.squared_difference(y, y_mean), 0)
    r2 = tf.subtract(tf.ones_like(num), tf.div(num, den))
    return tf.reduce_mean(r2)


def tf_safe_div(numerator, denominator):
    return tf.select(
        tf.greater(denominator, 0),
        tf.div(numerator,
               tf.select(
                   tf.equal(denominator, 0),
                   tf.ones_like(denominator),
                   denominator)),
        tf.zeros_like(numerator))


def tf_masked_r2_score(y_est, y_masked):
    y = y_masked[:, :, 0]
    y_est.get_shape().assert_is_compatible_with(y.get_shape())
    y_mask = y_masked[:, :, 1]
    y_sum = tf.reduce_sum(y, 0)
    y_count = tf.reduce_sum(y_mask, 0)
    y_mean = tf_safe_div(y_sum, y_count)
    num = tf.reduce_sum(tf.multiply(tf.squared_difference(y, y_est), y_mask), 0)
    den = tf.reduce_sum(tf.multiply(tf.squared_difference(y, y_mean), y_mask), 0)
    r2 = tf.subtract(tf.ones_like(num), tf_safe_div(num, den))
    return tf.reduce_mean(r2)
