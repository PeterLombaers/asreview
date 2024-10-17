from abc import ABC
from abc import abstractmethod

from asreview.data.record import Record
from asreview.data.utils import convert_to_list


class BaseReader(ABC):
    # When a data reader reads a file and turns it into records, it needs to know
    # which columns of the input data to put into which columns of the record. By
    # default these should be the same, but you can allow for alternative input column
    # names. For example, ASReview allows both 'title' or 'primary_title' for the
    # title column. The format is {record_column_name: [list of input column names]},
    # where the list of input column names is in order from most important to least
    # important. So when the input dataset contains two possible input columns for a
    # record column, it will pick the first it finds in the list.
    # If a field is not in this mapping, only the record column is allowed as input
    # column.
    __alternative_column_names__ = {
        "abstract": ["abstract", "notes_abstract", "abstract note"],
        "authors": ["authors", "first_authors", "author names"],
        "included": [
            "included",
            "label",
            "final_included",
            "label_included",
            "included_label",
            "included_final",
            "included_flag",
            "include",
        ],
        "title": ["title", "primary_title"],
    }

    __cleaning_methods__ = {
        "authors": convert_to_list,
        "keywords": convert_to_list,
    }

    @classmethod
    def read_records(cls, fp, dataset_id, record_class=Record, *args, **kwargs):
        df = cls.read_dataframe(fp, *args, **kwargs)
        print(df)
        df = cls.clean_data(df)
        return cls.to_records(df, dataset_id=dataset_id, record_class=record_class)

    @classmethod
    @abstractmethod
    def read_dataframe(cls, fp, *args, **kwargs):
        """Read data from a file into a Pandas dataframe.

        Anyone implementing a data reader class should inherit this base class. The main
        method to implement is `read_data`, which should produce a Pandas dataframe. The
        base class provides methods for cleaning the data and turning it into records.

        Parameters
        ----------
        fp : Path
            Filepath of the file to read.

        Returns
        -------
        pd.DataFrame
            A dataframe of user input data that has not been cleaned yet.
        """
        raise NotImplementedError

    @classmethod
    def clean_data(cls, df):
        df = cls.convert_alternative_column_names(df)
        for column, cleaning_method in cls.__cleaning_methods__.items():
            if column in df.columns:
                df[column] = df[column].apply(cleaning_method)
        return df

    @classmethod
    def to_records(cls, df, dataset_id=None, record_class=Record):
        columns_present = set(df.columns).intersection(set(record_class.get_columns()))
        return [
            record_class(dataset_row=idx, dataset_id=dataset_id, **row)
            for idx, row in df[list(columns_present)].iterrows()
        ]

    # Cleaning methods.
    @classmethod
    def convert_alternative_column_names(cls, df):
        """For record columns with alternative names, use the first available column.
        """
        for column, alternative_columns in cls.__alternative_column_names__.items():
            if column in df.columns:
                continue
            for alternative_column in alternative_columns:
                if alternative_column in df.columns:
                    df[column] = df[alternative_column]
                    break
        return df
