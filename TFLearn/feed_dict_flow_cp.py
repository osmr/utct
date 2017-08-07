from tflearn import data_flow


class FeedDictFlowCp(data_flow.FeedDictFlow):
    """
    Wrapper of TFLearn's FeedDictFlow for some types of augmentation.
    """

    def __init__(self,
                 feed_dict,
                 coord,
                 batch_size=128,
                 num_threads=8,
                 max_queue=32,
                 shuffle=False,
                 continuous=False,
                 ensure_data_order=False,
                 dprep_dict=None,
                 daug_dict=None,
                 index_array=None):

        super(FeedDictFlowCp, self).__init__(feed_dict,
                                             coord,
                                             batch_size,
                                             num_threads,
                                             max_queue,
                                             shuffle,
                                             continuous,
                                             ensure_data_order,
                                             dprep_dict,
                                             daug_dict,
                                             index_array)


    def fill_feed_dict_queue(self):

        while not self.coord.should_stop() and not self.interrupted:
            batch_ids = self.batch_ids_queue.get()
            if batch_ids is False:
                break
            data = self.retrieve_data(batch_ids)
            # Apply augmentation according to daug dict
            if self.daug_dict:
                for k in self.daug_dict:
                    data = self.daug_dict[k].apply(data)
            # Apply preprocessing according to dprep dict
            if self.dprep_dict:
                for k in self.dprep_dict:
                    data[k] = self.dprep_dict[k].apply(data[k])
            # all prepped, put the data into the queue
            self.feed_dict_queue.put(data)