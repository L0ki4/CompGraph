import typing as tp
from . import operations as ops
from .external_sort import ExternalSort


class Graph:
    """Computational graph implementation"""
    __slots__ = ("generator_name", "input_type", "file_name", "parser", "file_fabric", "operations_lst")

    def __init__(self, operations_lst: tp.List[tp.Any]):
        self.operations_lst = operations_lst

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        :param name: name of kwarg to use as data source
        """
        output_graph = Graph([])
        output_graph.generator_name = name  # type: ignore
        output_graph.input_type = "generator"  # type: ignore

        return output_graph

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        :param filename: filename to read from
        :param parser: parser from string to Row
        """

        def fabric() -> tp.Callable[[str, tp.Callable[[str], ops.TRow]], ops.TRowsGenerator]:
            def generator(filename: str, parser: tp.Callable[[str], ops.TRow]) -> ops.TRowsGenerator:
                with open(filename, "r") as f:
                    for str_ in f:
                        yield parser(str_)

            return generator

        output_graph = Graph([])
        output_graph.input_type = "file"  # type: ignore
        output_graph.file_name = filename  # type: ignore

        output_graph.parser = parser  # type: ignore
        output_graph.file_fabric = fabric()  # type: ignore
        return output_graph

    def copy(self) -> 'Graph':
        """Return copy of self"""
        output_graph = Graph([])
        for key in self.__slots__:
            try:
                attr_value = self.__getattribute__(key)
            except AttributeError:
                pass
            else:
                output_graph.__setattr__(key, attr_value)

        return output_graph

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        output_graph = self.copy()
        output_graph.operations_lst = output_graph.operations_lst + [ops.Map(mapper)]
        return output_graph

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        output_graph = self.copy()
        output_graph.operations_lst = output_graph.operations_lst + [ops.Reduce(reducer, keys)]
        return output_graph

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        output_graph = self.copy()
        output_graph.operations_lst = output_graph.operations_lst + [ExternalSort(keys)]
        return output_graph

    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """

        output_graph = self.copy()
        output_graph.operations_lst = output_graph.operations_lst + [(ops.Join(joiner, keys), join_graph)]
        return output_graph

    def run(self, **kwargs: tp.Any) -> tp.Union[tp.List[ops.TRow], ops.TRowsIterable]:
        """Single method to start execution; data sources passed as kwargs"""

        if self.input_type == "file":  # type: ignore
            row_iterator_creator = self.file_fabric  # type: ignore
            row_iterator = row_iterator_creator(self.file_name, self.parser)  # type: ignore
        else:
            row_iterator_creator = kwargs[self.generator_name]  # type: ignore
            row_iterator = row_iterator_creator()

        output_lst: tp.List[ops.TRowsGenerator] = [row_iterator]
        for i, func in enumerate(self.operations_lst):
            args = []
            if isinstance(func, tuple):  # operation is joi
                send_kwargs = kwargs.copy()
                send_kwargs["return_lst"] = False

                graph2join = func[1].run(**send_kwargs)
                func = func[0]
                args.append(graph2join)

            result = func(output_lst[i], *args)
            output_lst.append(result)

        if kwargs.get("return_lst", True):
            return list(output_lst[-1])
        else:
            return output_lst[-1]
