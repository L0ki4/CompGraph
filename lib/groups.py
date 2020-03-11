import typing as tp

TRow = tp.Dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]
TGroupGenerator = tp.Generator[TRowsIterable, None, None]


class GroupsCreator:
    """Class used to unite rows into groups with same values of keys"""

    def __init__(self, rows: TRowsIterable, keys: tp.Sequence[str]):
        """
        :param rows: table rows
        :param keys: keys to unite rows
        """

        self.keys = keys
        self.rows_generator = (row for row in rows)

        self.group_key_values: tp.Tuple[tp.Any, ...] = tuple()
        self.first_group_element: tp.Optional[TRow] = None

        try:
            self.first_group_element = next(self.rows_generator)
            self.group_key_values = tuple(self.first_group_element[key] for key in self.keys)
        except StopIteration:
            pass

        self.group_generator = self.group_iterator()

    def update_generator(self) -> None:
        """
        Exhaust generator of current group and update it with generator of new group
        """
        for _ in self.group_generator:
            pass

        self.group_generator = self.group_iterator()

    def group_iterator(self) -> TRowsGenerator:
        """
        Construct generator for new group if group exists
        """
        if self.first_group_element is None:
            return

        yield self.first_group_element

        for row in self.rows_generator:
            current_values = tuple(row[key] for key in self.keys)

            if current_values != self.group_key_values:
                self.first_group_element = row
                self.group_key_values = current_values
                return

            yield row

        self.first_group_element = None
