from abc import abstractmethod, ABC
import typing as tp
import string
from heapq import nlargest, nsmallest
from .groups import GroupsCreator
import math
from math import sin, cos, sqrt, atan2, radians
from datetime import datetime as dt

TRow = tp.Dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]
TGroupGenerator = tp.Generator[TRowsIterable, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    """Map class"""

    def __init__(self, mapper: Mapper) -> None:
        """
        :param mapper: mapper function
        """
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        """
        Construct map operation result generator
        :param rows: table rows
        """
        for row in rows:
            for result_row in self.mapper(row):
                yield result_row


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    """Reduce class"""

    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        """
        :param reducer: reducer function
        :param keys: keys to unite rows in groups.py
        """
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        """
        Construct reduce operation result generator
        :param rows: table rows
        """
        groups_creator = GroupsCreator(rows, self.keys)

        while groups_creator.first_group_element is not None:
            for result_row in self.reducer(tuple(self.keys), groups_creator.group_generator):
                yield result_row

            groups_creator.update_generator()


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = "_1", suffix_b: str = "_2") -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


def compare_key_values(first_values: tp.Tuple[tp.Any, ...],
                       second_values: tp.Tuple[tp.Any, ...]) -> tp.Tuple[int, int]:
    """
    Lexicographically compare two dicts of key values
    :param first_values: dict of key values
    :param second_values: dict of key values
    :return return 0 if values are equal, 1 if first_values is bigger, -1 otherwise
    """
    for first, second in zip(first_values, second_values):
        if first < second:
            return 0, 1
        elif first > second:
            return 1, 0

    return 1, 1


class Join(Operation):
    """Join class"""

    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        """
        :param joiner: join strategy
        :param keys: keys to unite rows in groups.py
        """
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        """
        Construct join operation result generator
        :param rows: left table rows
        :param args: [right table rows]
        """
        left_groups_creator = GroupsCreator(rows, self.keys)
        right_groups_creator = GroupsCreator(args[0], self.keys)

        while left_groups_creator.first_group_element and right_groups_creator.first_group_element:
            left_group_operand, right_group_operand = compare_key_values(left_groups_creator.group_key_values,
                                                                         right_groups_creator.group_key_values)

            if left_group_operand == right_group_operand:
                for result_row in self.joiner(self.keys, left_groups_creator.group_generator,
                                              right_groups_creator.group_generator):
                    yield result_row

                left_groups_creator.update_generator()
                right_groups_creator.update_generator()

            elif left_group_operand > right_group_operand:
                for result_row in self.joiner(self.keys, [dict()], right_groups_creator.group_generator):
                    yield result_row

                right_groups_creator.update_generator()

            else:
                for result_row in self.joiner(self.keys, left_groups_creator.group_generator, [dict()]):
                    yield result_row

                left_groups_creator.update_generator()

        while left_groups_creator.first_group_element is not None:
            for result_row in self.joiner(self.keys, left_groups_creator.group_generator, [dict()]):
                yield result_row

            left_groups_creator.update_generator()

        while right_groups_creator.first_group_element is not None:
            for result_row in self.joiner(self.keys, [dict()], right_groups_creator.group_generator):
                yield result_row

            right_groups_creator.update_generator()


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        string_ = row.get(self.column, "")
        row[self.column] = string_.translate(str.maketrans("", "", string.punctuation))
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self._lower_case(row.get(self.column, ""))
        yield row


class InverseFrequency(Mapper):
    """inverse frequency """

    def __init__(self, elements_amount_column: str, encountered_elements_amount_column: str, res_row: str = "idf"):
        """
        :param elements_amount_column: name of column with elements amount
        :param encountered_elements_amount_column: name of column with overall elements amount
        :param res_row: name of column to store the result
        """
        self.elements_amount_column = elements_amount_column
        self.encountered_elements_amount_column = encountered_elements_amount_column
        self.res_row = res_row

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.res_row] = math.log(row[self.elements_amount_column] / row[self.encountered_elements_amount_column])
        yield row


class CalculateDistance(Mapper):
    """Calculate distance by coordinates in kilometres"""

    def __init__(self, first_coordinate: str, second_coordinate: str, distance_column: str):
        """
        :param first_coordinate: name of column with start coordinate, format: (lon, lat)
        :param second_coordinate: name of column with end coordinate, format: (lon, lat)
        :param distance_column: name of column to store the result
        """
        self.first_coordinate = first_coordinate
        self.second_coordinate = second_coordinate
        self.distance_column = distance_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        longitude_first = radians(row[self.first_coordinate][0])
        latitude_first = radians(row[self.first_coordinate][1])

        longitude_second = radians(row[self.second_coordinate][0])
        latitude_second = radians(row[self.second_coordinate][1])

        longitude_delta = longitude_second - longitude_first
        latitude_delta = latitude_second - latitude_first

        square_sin_latitude = sin(latitude_delta / 2) ** 2
        square_sin_longitude = sin(longitude_delta / 2) ** 2
        trigonometry_term = square_sin_latitude + cos(latitude_first) * cos(latitude_second) * square_sin_longitude

        earth_radius = 6373.0
        row[self.distance_column] = 2 * earth_radius * atan2(sqrt(trigonometry_term), sqrt(1 - trigonometry_term))
        yield row


class WeekDay(Mapper):
    """Get day of the week by date"""

    def __init__(self, date_column: str, week_day_column: str):
        """
        :param date_column: name of column with date
        :param week_day_column: name of column to store the result
        """
        self.date_column = date_column
        self.week_day_column = week_day_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        try:
            row[self.week_day_column] = dt.strptime(row[self.date_column], "%Y%m%dT%H%M%S.%f").strftime("%A")[:3]
        except ValueError:
            row[self.week_day_column] = dt.strptime(row[self.date_column], "%Y%m%dT%H%M%S").strftime("%A")[:3]
        yield row


class Hour(Mapper):
    """Get hour by date"""

    def __init__(self, date_column: str, hour_column: str):
        """
        :param date_column: name of column with date
        :param hour_column: name of column to store the result
        """
        self.date_column = date_column
        self.hour_column = hour_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        try:
            row[self.hour_column] = dt.strptime(row[self.date_column], "%Y%m%dT%H%M%S.%f").hour
        except ValueError:
            row[self.hour_column] = dt.strptime(row[self.date_column], "%Y%m%dT%H%M%S").hour
        yield row


class TimeDelta(Mapper):
    """Calculate time delta between dates in seconds"""

    def __init__(self, start_date_col: str, end_date_col: str, time_delta_column: str):
        """
        :param start_date_col: name of column with start date
        :param end_date_col: name of column with end date
        :param time_delta_column: name of column to store the result
        """
        self.start_date_col = start_date_col
        self.end_date_col = end_date_col
        self.time_delta_column = time_delta_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        try:
            enter_time = dt.strptime(row[self.start_date_col], "%Y%m%dT%H%M%S.%f")
        except ValueError:
            enter_time = dt.strptime(row[self.start_date_col], "%Y%m%dT%H%M%S")

        try:
            exit_time = dt.strptime(row[self.end_date_col], "%Y%m%dT%H%M%S.%f")
        except ValueError:
            exit_time = dt.strptime(row[self.end_date_col], "%Y%m%dT%H%M%S")

        row[self.time_delta_column] = (exit_time - enter_time).total_seconds()
        yield row


class Speed(Mapper):
    """Calculate speed in kilometres per hour"""

    def __init__(self, distance_column: str, time_column: str, speed_column: str):
        """
        :param distance_column: name of column with distance
        :param time_column: name of column with time
        :param speed_column: name of column to store the result
        """
        self.distance_column = distance_column
        self.time_column = time_column
        self.speed_column = speed_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.speed_column] = row[self.distance_column] / row[self.time_column] * (60 ** 2)
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: tp.Optional[str] = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        str2split = row.get(self.column, "")
        for substr in str2split.split(self.separator):
            new_row = row.copy()
            new_row[self.column] = substr
            yield new_row


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = "product") -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_column] = 1

        for column in self.columns:
            row[self.result_column] *= row[column]
        yield row


class Filter(Mapper):
    """Remove records that don"t satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        result_row = {}
        for column in self.columns:
            result_row[column] = row[column]

        yield result_row


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int, ascending: bool = True) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n
        self.ascending = ascending

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        if self.ascending:
            for row in nlargest(self.n, rows, key=lambda dict_: dict_[self.column_max]):
                yield row
        else:
            for row in nsmallest(self.n, rows, key=lambda dict_: dict_[self.column_max]):
                yield row


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = "tf") -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_keys: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        words_amount = 0
        words_counter: tp.Dict[str, int] = dict()

        for row in rows:
            words_amount += 1
            words_counter.setdefault(row[self.words_column], 0)
            words_counter[row[self.words_column]] += 1

        common_keys: TRow = dict()
        for key in group_keys:
            try:
                common_keys[key] = row[key]
            except NameError:
                raise NameError('Empty row iterator!')

        for key in words_counter:
            result = common_keys.copy()
            result[self.words_column] = key
            result[self.result_column] = words_counter[key] / words_amount
            yield result


class Count(Reducer):
    """Count rows passed and yield single row as a result"""

    def __init__(self, column: str) -> None:
        """
        :param column: name of column to count
        """
        self.column = column

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        counter = 0
        for row in rows:
            counter += 1

        ans: TRow = dict()
        for key in group_key:
            try:
                ans[key] = row[key]
            except NameError:
                raise NameError('Empty row iterator!')

        ans[self.column] = counter
        yield ans


class Sum(Reducer):
    """Sum values in column passed and yield single row as a result"""

    def __init__(self, column: str) -> None:
        """
        :param column: name of column to sum
        """
        self.column = column

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        sum_ = 0
        for row in rows:
            sum_ += row[self.column]

        try:
            ans = {group_key[0]: row[group_key[0]], self.column: sum_}
        except NameError:
            raise NameError('Empty row iterator!')

        yield ans


class Mean(Reducer):
    """Mean values in column passed and yield single row as a result"""

    def __init__(self, column: str) -> None:
        """
        :param column: name of column to mean
        """
        self.column = column

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        sum_ = 0
        amount = 0
        for row in rows:
            sum_ += row[self.column]
            amount += 1

        try:
            ans = dict()
            for key in group_key:
                ans[key] = row[key]
            ans[self.column] = sum_ / amount
        except NameError:
            raise NameError('Empty row iterator!')

        yield ans


# Joiners

def merge_two_dicts_by_keys(a_row: TRow, b_row: TRow, suffix_a: str, suffix_b: str, keys: tp.Sequence[str]) -> TRow:
    """
    Merge two rows with same values in columns
    :param a_row: row to merge
    :param b_row: row to merge
    :param suffix_a: added while merge to column name form a_row if dicts has same column names not from keys
    :param suffix_b: added while merge to column name from b_row if dicts has same column names not from keys
    :param keys: column names to merge by

    :return return row with columns from a_row and b_row
    """

    # empty row always in b_row
    if len(a_row) == 0:
        a_row = b_row
        b_row = dict()

        suf_bank = suffix_a
        suffix_a = suffix_b
        suffix_b = suf_bank

    merged_dct = dict()

    for key in a_row:
        if key in b_row and key not in keys:
            merged_dct[key + suffix_a] = a_row[key]
            merged_dct[key + suffix_b] = b_row[key]
        else:
            merged_dct[key] = a_row[key]

    for key in b_row:
        if key not in merged_dct and key + suffix_b not in merged_dct:
            merged_dct[key] = b_row[key]

    return merged_dct


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        left_dict_is_none = rows_a == [dict()]
        right_dict_is_none = rows_b == [dict()]

        rows_b = tuple(rows_b)

        for row_a in rows_a:
            for row_b in rows_b:
                if not left_dict_is_none and not right_dict_is_none:
                    yield merge_two_dicts_by_keys(row_a, row_b, self._a_suffix, self._b_suffix, keys)


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows_b = tuple(rows_b)

        for row_a in rows_a:
            for row_b in rows_b:
                yield merge_two_dicts_by_keys(row_a, row_b, self._a_suffix, self._b_suffix, keys)


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        left_dict_is_none = rows_a == [dict()]
        rows_b = tuple(rows_b)

        for row_a in rows_a:
            for row_b in rows_b:
                if not left_dict_is_none:
                    yield merge_two_dicts_by_keys(row_a, row_b, self._a_suffix, self._b_suffix, keys)


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        right_dict_is_none = rows_b == [dict()]
        rows_b = tuple(rows_b)

        for row_a in rows_a:
            for row_b in rows_b:
                if not right_dict_is_none:
                    yield merge_two_dicts_by_keys(row_a, row_b, self._a_suffix, self._b_suffix, keys)
