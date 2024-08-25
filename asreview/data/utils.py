import json
from ast import literal_eval
import numpy as np
import pandas as pd


def duplicated(df, pid="doi"):
    """Return boolean Series denoting duplicate rows.

    Identify duplicates based on titles and abstracts and if available,
    on a persistent identifier (PID) such as the Digital Object Identifier
    (`DOI <https://www.doi.org/>`_).

    Arguments
    ---------
    df : pd.DataFrame or DataStore
        Dataframe containing columns 'title', 'abstract' and optionally a column
        containing identifiers of type `pid`.
    pid: string
        Which persistent identifier to use for deduplication.
        Default is 'doi'.

    Returns
    -------
    pandas.Series
        Boolean series for each duplicated rows.
    """
    if pid in df.columns:
        # in case of strings, strip whitespaces and replace empty strings with None
        if pd.api.types.is_string_dtype(df[pid]) or pd.api.types.is_object_dtype(
            df[pid]
        ):
            s_pid = df[pid].str.strip().replace("", None)
            if pid == "doi":
                s_pid = s_pid.str.lower().str.replace(
                    r"^https?://(www\.)?doi\.org/", "", regex=True
                )
        else:
            s_pid = df[pid]

        # save boolean series for duplicates based on persistent identifiers
        s_dups_pid = (s_pid.duplicated()) & (s_pid.notnull())
    else:
        s_dups_pid = None

    # get the texts, clean them and replace empty strings with None
    s = (
        pd.Series(get_texts(df))
        .str.replace("[^A-Za-z0-9]", "", regex=True)
        .str.lower()
        .str.strip()
        .replace("", None)
    )

    # save boolean series for duplicates based on titles/abstracts
    s_dups_text = (s.duplicated()) & (s.notnull())

    # final boolean series for all duplicates
    if s_dups_pid is not None:
        s_dups = s_dups_pid | s_dups_text
    else:
        s_dups = s_dups_text

    return s_dups


def get_texts(df):
    """Get texts from a dataframe containing at least one of 'title' and 'abstract'.

    A text consists of the title and abstract concatenated."""
    # One of title and abstract is always present.
    try:
        titles = df["title"]
    except KeyError:
        return df["abstract"]
    try:
        abstracts = df["abstract"]
    except KeyError:
        return df["title"]

    s_title = pd.Series(titles).fillna("")
    s_abstract = pd.Series(abstracts).fillna("")

    cur_texts = (s_title + " " + s_abstract).str.strip()

    return cur_texts.values


def convert_keywords(keywords):
    """Split keywords separated by commas etc to lists."""
    if not isinstance(keywords, str):
        return keywords

    current_best = [keywords]
    for splitter in [", ", "; ", ": ", ",", ";", ":"]:
        new_split = keywords.split(splitter)
        if len(new_split) > len(current_best):
            current_best = new_split
    return current_best


def convert_to_list(value):
    """Convert a value to a list.

    This function tries to be very permissive in what input it allows. The goal is to
    accept input from as many different kinds of input files as possible. If you are
    certain what format the input has, you are probably better of parsing that format
    directly.
    """
    if isinstance(value, list):
        return value
    elif isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        if value == "":
            return []
        if value[0] == "[":
            # Check if it's a json dumped list.
            try:
                return json.loads(value)
            except json.decoder.JSONDecodeError:
                # Maybe it's a Python literal value?
                try:
                    return literal_eval(value)
                except SyntaxError:
                    raise ValueError(
                        "value is a string starting with '[', but is not a JSON dumped"
                        f" list or a Python literal list value. Value: {value}"
                    )
        # Assume it is a list of items separated by one of ,;:
        longest_split = []
        for sep in {",", ";", ":"}:
            split_value = value.split(sep)
            if len(split_value) > len(longest_split):
                longest_split = split_value
        # Remove excess whitespace in case the items were separated by ', ' for example.
        return [item.strip() for item in longest_split]
