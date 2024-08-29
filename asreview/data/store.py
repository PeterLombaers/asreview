import numpy as np
import pandas as pd
from sqlalchemy import NullPool
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import Session

from asreview.data.record import Base
from asreview.data.record import Record

CURRENT_DATASTORE_VERSION = 0


class DataStore:
    def __init__(self, fp, record_cls=Record):
        self.fp = fp
        # I'm using NullPool here, indicating that the engine should use a connection
        # pool, but just create and dispose of a connection every time a request comes.
        # This makes it very easy dispose of the engine, but is less efficient.
        # I was getting errors when running tests that try to clean up behind them,
        # and this solves those errors. We can change this back to a connection pool at
        # some later moment by properly looking at how to close everything.
        self.engine = create_engine(f"sqlite+pysqlite:///{self.fp}", poolclass=NullPool)
        self.record_cls = record_cls
        self._columns = self.record_cls.get_columns()
        self._pandas_dtype_mapping = self.record_cls.get_pandas_dtype_mapping()

    @property
    def columns(self):
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

    def add_records(self, records):
        """Add records to the data store.

        Parameters
        ----------
        records : list[self.record_cls]
            List of records to add to the store.
        """
        with Session(self.engine) as session:
            session.add_all(records)
            session.commit()

    def __len__(self):
        with Session(self.engine) as session:
            return session.query(self.record_cls).count()

    def __getitem__(self, item):
        # We allow a string or a list of strings as input. If the input is a string we
        # return that column as a pandas series. If the input is a list of strings we
        # return a pandas DataFrame containing those columns. This way the output you
        # get is the same if you do __getitem__ on a DataStore instance or on a pandas
        # DataFrame containing the same data.
        if isinstance(item, str):
            columns = [item]
        else:
            columns = item
        with self.engine.connect() as con:
            df = pd.read_sql(
                self.record_cls.__tablename__,
                con,
                columns=columns,
                dtype=self.pandas_dtype_mapping,
            )
        if isinstance(item, str):
            return df[item]
        else:
            return df

    def __contains__(self, item):
        return item in self.columns

    def is_empty(self):
        with Session(self.engine) as session:
            return session.query(self.record_cls).first() is None

    def get_records(self, record_id):
        """Get the records with the given record identifiers.

        Arguments
        ---------
        record_id : int | list[int]
            Record id or list of record id's of records to get.

        Returns
        -------
        asreview.data.record.Record or list of records.
        """
        if isinstance(record_id, np.integer):
            record_id = record_id.item()

        with Session(self.engine) as session:
            if isinstance(record_id, int):
                return (
                    session.query(self.record_cls)
                    .filter(self.record_cls.record_id == record_id)
                    .first()
                )
            else:
                return (
                    session.query(self.record_cls)
                    .filter(self.record_cls.record_id.in_(record_id))
                    .all()
                )

    def get_df(self):
        """Get all data from the data store as a pandas DataFrmae.

        Returns
        -------
        pd.DataFrame
        """
        with self.engine.connect() as con:
            return pd.read_sql(
                self.record_cls.__tablename__,
                con,
                dtype=self.pandas_dtype_mapping,
            )
