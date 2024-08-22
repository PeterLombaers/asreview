import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import Session

from asreview.data.base import Dataset
from asreview.data.record import Base
from asreview.data.record import Record


CURRENT_DATASTORE_VERSION = 0


class DataStore:
    def __init__(self, fp, record_cls=Record):
        self.fp = fp
        self.engine = create_engine(f"sqlite+pysqlite:///{self.fp}")
        self.record_cls = record_cls
        self._columns = self.record_cls.get_columns()
        self._pandas_dtype_mapping = self.record_cls.get_pandas_dtype_mapping()

    @property
    def columns(self):
        """"""
        return self._columns

    @property
    def pandas_dtype_mapping(self):
        return self._pandas_dtype_mapping

    @property
    def user_version(self):
        """Version number of the state."""
        with self.engine.connect() as conn:
            version = conn.execute(text("PRAGMA user_version"))
            return int(version.fetchone()[0])

    @user_version.setter
    def user_version(self, version):
        with self.engine.connect() as conn:
            # I tried passing the version through a parameter, but it didn't seem to
            # work. Maybe you can't use parameters with PRAGMA statements?
            conn.execute(text(f"PRAGMA user_version = {version}"))
            conn.commit()

    def create_tables(self):
        """Initialize the tables containing the data."""
        self.user_version = CURRENT_DATASTORE_VERSION
        Base.metadata.create_all(self.engine)

    def add_dataset(self, dataset, dataset_id):
        """Add a new dataset to the data store."""
        # try:
        #     keywords = [
        #         json.dumps(keyword_list) for keyword_list in dataset["keywords"]
        #     ]
        # except KeyError:
        #     keywords = None
        records = dataset.record(range(len(dataset)))
        for record in records:
            record.dataset_id = dataset_id
        with Session(self.engine) as session:
            session.add_all(records)
            session.commit()

    def __len__(self):
        with Session(self.engine) as session:
            return session.query(self.record_cls).count()

    def __getitem__(self, item):
        if isinstance(item, str):
            item = [item]
        return pd.read_sql(
            self.record_cls.__tablename__,
            self.engine.connect(),
            columns=item,
            dtype=self.pandas_dtype_mapping,
        )

    def __contains__(self, item):
        return item in self.columns

    def is_empty(self):
        with Session(self.engine) as session:
            return session.query(self.record_cls).first() is None

    def get_record(self, record_id):
        """Get the record with the given record row number.

        Arguments
        ---------
        record_id : int
            Record id (row number) of the record to get.

        Returns
        -------
        asreview.data.record.Record
        """
        if isinstance(record_id, np.int64):
            record_id = record_id.item()

        with Session(self.engine) as session:
            if isinstance(record_id, int):
                return (
                    session.query(self.record_cls)
                    .filter(self.record_cls.id == record_id)
                    .first()
                )
            else:
                return session.query(self.record_cls).filter(
                    self.record_cls.id.in_(record_id)
                )

    def get_all(self):
        """Get all data from the data store as a dataset.

        Returns
        -------
        asreview.data.base.Dataset
        """
        df = pd.read_sql(
            self.record_cls.__tablename__,
            self.engine.connect(),
            dtype=self.pandas_dtype_mapping,
        )
        return Dataset(df=df)
