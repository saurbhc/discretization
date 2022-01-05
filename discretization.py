from math import log2

import pandas as pd


def calculate_log2(input):
    if input == 0:
        return 0

    return log2(input)


class Discretization:
    def __init__(self, dataset_file_name="dataset.csv"):
        self.dataset_name = dataset_file_name
        self.dataset = None
        self.information_gain_of_whole_dataset = None
        self.splitting_points = None
        self._greatest_gain_in_entropy = -1
        self._best_split_dict = None

    def execute(self):
        self._import_dataset()
        self._calculate_information_gain_of_whole_dataset()
        self._evaluate_splitting_points()
        self._compute_entropy_gains_of_all_splits()
        print(
            "Best Split was on splitting point: ",
            self._best_split_dict["splitting_point"],
            " with an information-gain of: ",
            self._best_split_dict["_information_gain_for_current_split"],
            " and highest entropy of: ",
            self._best_split_dict["_net_entropy_for_current_split"],
        )
        print(
            "Best Split lte-data-points: ",
            self._best_split_dict["_lte_split_data_points"],
        )
        print(
            "Best Split gt_split_data_points: ",
            self._best_split_dict["_gt_split_data_points"],
        )

        return self._best_split_dict["splitting_point"]

    def _import_dataset(self):
        self.dataset = pd.read_csv(self.dataset_name, sep=",", header=0)

    def _calculate_information_gain_of_whole_dataset(self):
        unique_labels = self.dataset.label.unique()
        number_of_all_data_points = self.dataset.data_points.count()
        all_probabilities = []

        for unique_label in unique_labels:
            number_of_unique_label_data_points = (
                self.dataset.where(self.dataset["label"] == unique_label)
                .dropna()
                .data_points.count()
            )

            probability_of_unique_label = (
                number_of_unique_label_data_points / number_of_all_data_points
            )
            all_probabilities.append(probability_of_unique_label)

        all_probabilities_with_their_log = []
        for _probability in all_probabilities:
            all_probabilities_with_their_log.append(
                _probability * calculate_log2(_probability)
            )

        self.information_gain_of_whole_dataset = -1 * sum(
            all_probabilities_with_their_log
        )
        print(
            "information_gain_of_whole_dataset: ",
            self.information_gain_of_whole_dataset,
        )

    def _evaluate_splitting_points(self):
        self.splitting_points = []

        _last_data_point = None
        for _data_point in self.dataset.data_points.to_list():
            if not _last_data_point:
                _last_data_point = _data_point
                continue

            self.splitting_points.append(
                {
                    "splitting_point": (_last_data_point + _data_point) / 2,
                    "data_points": [_last_data_point, _data_point],
                }
            )

            _last_data_point = _data_point
        print("splitting_points: ", self.splitting_points)

    def _compute_entropy_gains_of_all_splits(self):
        for _splitting_point_dict in self.splitting_points:
            _splitting_point = _splitting_point_dict["splitting_point"]
            print("_splitting_point: ", _splitting_point)

            _less_than_equal_to_splitting_point_df = self.dataset.where(
                self.dataset["data_points"] <= _splitting_point
            ).dropna()

            _greater_than_splitting_point_df = self.dataset.where(
                self.dataset["data_points"] > _splitting_point
            ).dropna()

            _all_probabilities_for_less_than_equal_to_with_their_log = []
            _all_probabilities_for_greater_than_with_their_log = []
            for _label in self.dataset.label.unique():
                # compute probability for l.t.e
                df_lte = _less_than_equal_to_splitting_point_df.where(
                    _less_than_equal_to_splitting_point_df["label"] == _label
                )
                p_lte = (
                    df_lte.data_points.count()
                    / _less_than_equal_to_splitting_point_df.data_points.count()
                )
                _all_probabilities_for_less_than_equal_to_with_their_log.append(
                    p_lte * calculate_log2(p_lte)
                )

                # compute probability for g.t
                df_lte = _greater_than_splitting_point_df.where(
                    _greater_than_splitting_point_df["label"] == _label
                )
                p_gt = (
                    df_lte.data_points.count()
                    / _greater_than_splitting_point_df.data_points.count()
                )
                _all_probabilities_for_greater_than_with_their_log.append(
                    p_gt * calculate_log2(p_gt)
                )

            _entropy_for_lte = -1 * sum(
                _all_probabilities_for_less_than_equal_to_with_their_log
            )
            _entropy_for_gt = -1 * sum(
                _all_probabilities_for_greater_than_with_their_log
            )

            _probability_used_in_entropy_for_lte = (
                _less_than_equal_to_splitting_point_df.data_points.count()
                / self.dataset.data_points.count()
            )
            _probability_used_in_entropy_for_ge = (
                _greater_than_splitting_point_df.data_points.count()
                / self.dataset.data_points.count()
            )

            _net_entropy_for_current_split = (
                _probability_used_in_entropy_for_lte * _entropy_for_lte
            ) + (_probability_used_in_entropy_for_ge * _entropy_for_gt)

            _information_gain_for_current_split = (
                self.information_gain_of_whole_dataset - _net_entropy_for_current_split
            )
            print(
                "..._information_gain_for_current_split: ",
                _information_gain_for_current_split,
            )

            if _information_gain_for_current_split > self._greatest_gain_in_entropy:
                self._greatest_gain_in_entropy = _information_gain_for_current_split
                _splitting_point_dict[
                    "_information_gain_for_current_split"
                ] = _information_gain_for_current_split
                _splitting_point_dict[
                    "_net_entropy_for_current_split"
                ] = _net_entropy_for_current_split
                _splitting_point_dict[
                    "_lte_split_data_points"
                ] = _less_than_equal_to_splitting_point_df.data_points.tolist()
                _splitting_point_dict[
                    "_gt_split_data_points"
                ] = _greater_than_splitting_point_df.data_points.tolist()
                self._best_split_dict = _splitting_point_dict


if __name__ == "__main__":
    d_obj = Discretization()
    d_obj.execute()
