import os


class Saver(object):
    """
    Class which calculate and accumulate auxiliary file paths for model training. Also it saves
    information into some log-files by requests.

    Parameters:
    ----------
    work_dirname : str
        directory path, where we can save all our data
    project_name : str
        name of subdirectory in the working directory, where we will save all our data for 
        current task
    model_filename_prefix : str
        preffix for models saved files
    log_filename : str
        file name for training log
    score_log_filename : str
        file name for table with scores, updated during training
    hyper_log_filename : str
        file name for table with scores, updated during hyper-optimization
    score_ref_filename : str
        file name for table of projects, existed in the working directory
    last_model_subdir_name : str
        directory name for models currently trained
    best_model_subdir_name : str
        directory name for models with the best scores
    log_subdir_name : str
        directory name for a third-party logger, e.g. for TensorBoard
    score_log_subdir_name : str
        another directory name for a third-party logger
    """

    def __init__(self,
                 work_dirname,
                 project_name,
                 model_filename_prefix,
                 log_filename=None,
                 score_log_filename=None,
                 hyper_log_filename=None,
                 score_ref_filename=None,
                 last_model_subdir_name="last_model",
                 best_model_subdir_name="best_model",
                 log_subdir_name="train_log",
                 score_log_subdir_name="score_train_log"):

        self.work_dirname = work_dirname
        if not os.path.exists(self.work_dirname):
            os.makedirs(self.work_dirname)

        self.project_name = project_name
        self.project_dirname = os.path.join(self.work_dirname, project_name)
        if not os.path.exists(self.project_dirname):
            os.makedirs(self.project_dirname)

        self.last_checkpoints_dirname = os.path.join(
            self.project_dirname, last_model_subdir_name)
        if not os.path.exists(self.last_checkpoints_dirname):
            os.makedirs(self.last_checkpoints_dirname)

        self.best_checkpoints_dirname = os.path.join(
            self.project_dirname, best_model_subdir_name)
        if not os.path.exists(self.best_checkpoints_dirname):
            os.makedirs(self.best_checkpoints_dirname)

        self.log_dirname = os.path.join(self.project_dirname, log_subdir_name)
        if not os.path.exists(self.log_dirname):
            os.makedirs(self.log_dirname)

        if (score_log_subdir_name is not None) and (len(score_log_subdir_name) > 0):
            self.score_log_dirname = os.path.join(
                self.project_dirname, score_log_subdir_name)
            if not os.path.exists(self.score_log_dirname):
                os.makedirs(self.score_log_dirname)
        else:
            self.score_log_dirname = self.project_dirname

        self.model_filename_prefix = model_filename_prefix
        self.update_log_filename(log_filename)
        self.update_score_log_filename(score_log_filename)
        self.update_hyper_log_filename(hyper_log_filename)
        self.update_score_ref_filename(score_ref_filename)

    def update_log_filename(self, log_filename):
        """
        Updating the file name for training log.

        Parameters:
        ----------
        log_filename : str
            new value for the file name for training log
        """
        self.log_filename = log_filename
        if (log_filename is not None) and (len(log_filename) > 0):
            self.log_filename = os.path.join(self.log_dirname, log_filename)

    def update_score_log_filename(self, score_log_filename):
        """
        Updating the file name for table with scores, updated during training.

        Parameters:
        ----------
        score_log_filename : str
            new value for the file name for table with scores, updated during training
        """
        self.score_log_filename = score_log_filename
        self.score_log_rel_filename = score_log_filename
        if (score_log_filename is not None) and (len(score_log_filename) > 0):
            self.score_log_filename = os.path.join(self.score_log_dirname, score_log_filename)
            self.score_log_rel_filename = os.path.relpath(self.score_log_filename, self.work_dirname)

    def update_hyper_log_filename(self, hyper_log_filename):
        """
        Updating the file name for table with scores, updated during hyper-optimization.

        Parameters:
        ----------
        hyper_log_filename : str
            new value for the file name for table with scores, updated during hyper-optimization
        """
        self.hyper_log_filename = hyper_log_filename
        self.hyper_log_rel_filename = hyper_log_filename
        if (hyper_log_filename is not None) and (len(hyper_log_filename) > 0):
            self.hyper_log_filename = os.path.join(self.project_dirname, hyper_log_filename)
            self.hyper_log_rel_filename = os.path.relpath(self.hyper_log_filename, self.work_dirname)
        self.hyper_log_ref_filename = self.hyper_log_rel_filename if (self.hyper_log_rel_filename is not None) else "NA"

    def update_score_ref_filename(self, score_ref_filename):
        """
        Updating the file name for table of projects, existed in the working directory.

        Parameters:
        ----------
        score_ref_filename : str
            new value for the file name for table of projects, existed in the working directory
        """
        self.score_ref_filename = score_ref_filename
        if (score_ref_filename is not None) and (len(score_ref_filename) > 0):
            self.score_ref_filename = os.path.join(self.work_dirname, score_ref_filename)
        score_ref_file_exist = os.path.exists(self.score_ref_filename) and (os.path.getsize(self.score_ref_filename) > 0)
        if score_ref_file_exist:
            import pandas as pd
            df = pd.read_csv(self.score_ref_filename, sep="\t", header=None, names=['project_name', 'score_log_filename', 'hyper_log_filename'])
            df_sub = df[df['project_name'] == self.project_name]
            if len(df_sub.index) > 1:
                raise Exception("Error: Score reference file has dublicated rows: {}".format(self.score_ref_filename))
            elif len(df_sub.index) == 0:
                self._add_line_to_score_ref_file()
            else:
                if (df_sub['score_log_filename'].iloc[0] != self.score_log_rel_filename) or (df_sub['hyper_log_filename'].iloc[0] != self.hyper_log_ref_filename):
                    raise Exception("Error: Score reference file has a wrong record: {}".format(self.score_ref_filename))
        else:
            self._add_line_to_score_ref_file()

    def _add_line_to_score_ref_file(self):
        """
        Write the appropriate record into the table of projects, existed in the working directory.
        """
        score_ref_file = open(self.score_ref_filename, "a")
        score_ref_file.write("\t".join([self.project_name, self.score_log_rel_filename, self.hyper_log_ref_filename]) + "\n")
        score_ref_file.close()
