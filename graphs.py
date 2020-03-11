from .lib import Graph, operations
import typing as tp


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count') -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    return Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def word_count_graph_from_file(input_stream_name: str, parser: tp.Callable[[str], operations.TRow],
                               text_column: str = 'text', count_column: str = 'count') -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    return Graph.graph_from_file(input_stream_name, parser) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf') -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    graph1 = Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    count_column = "docs_amount"
    graph2 = Graph.graph_from_iter(input_stream_name) \
        .sort([doc_column]) \
        .reduce(operations.Count(count_column), [])

    suffix_enc = ""
    suffix_all = "_overall"
    idf_graph = graph1.sort([doc_column, text_column]) \
        .reduce(operations.FirstReducer(), [doc_column, text_column]) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .join(operations.InnerJoiner(suffix_enc, suffix_all), graph2, []) \
        .map(operations.InverseFrequency(count_column + suffix_all, count_column + suffix_enc))

    tf_col = "tf"
    idf_col = "idf"
    result_graph = graph1.sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column), [doc_column]) \
        .sort([text_column]) \
        .join(operations.InnerJoiner(), idf_graph, [text_column]) \
        .map(operations.Product([tf_col, idf_col], result_column)) \
        .reduce(operations.TopN(result_column, 3), [text_column]) \
        .sort([doc_column]) \
        .map(operations.Project([doc_column, text_column, result_column]))

    return result_graph


def inverted_index_graph_from_file(input_stream_name: str, parser: tp.Callable[[str], operations.TRow],
                                   doc_column: str = 'doc_id', text_column: str = 'text',
                                   result_column: str = 'tf_idf') -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    graph1 = Graph.graph_from_file(input_stream_name, parser) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    count_column = "docs_amount"
    graph2 = Graph.graph_from_file(input_stream_name, parser) \
        .sort([doc_column]) \
        .reduce(operations.Count(count_column), [])

    suffix_enc = ""
    suffix_all = "_overall"
    idf_graph = graph1.sort([doc_column, text_column]) \
        .reduce(operations.FirstReducer(), [doc_column, text_column]) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .join(operations.InnerJoiner(suffix_enc, suffix_all), graph2, []) \
        .map(operations.InverseFrequency(count_column + suffix_all, count_column + suffix_enc))

    tf_col = "tf"
    idf_col = "idf"
    result_graph = graph1.sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column), [doc_column]) \
        .sort([text_column]) \
        .join(operations.InnerJoiner(), idf_graph, [text_column]) \
        .map(operations.Product([tf_col, idf_col], result_column)) \
        .reduce(operations.TopN(result_column, 3), [text_column]) \
        .sort([doc_column]) \
        .map(operations.Project([doc_column, text_column, result_column]))

    return result_graph


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi') -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    count_column = "words_amount"

    def words_filter(row: operations.TRow) -> bool:
        return len(row[text_column]) > 4 and row[count_column] > 1

    graph1 = Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([doc_column, text_column])

    filter_graph = graph1.reduce(operations.Count(count_column), [doc_column, text_column]) \
        .map(operations.Filter(words_filter))

    filtered_graph = graph1.join(operations.InnerJoiner(), filter_graph, [doc_column, text_column])

    frequency_column = "words_frequency"
    graph2 = filtered_graph.reduce(operations.TermFrequency(text_column, frequency_column), []) \
        .sort([text_column])

    suffix_enc = ""
    suffix_all = "_overall"

    graph3 = filtered_graph.sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column, frequency_column), [doc_column]) \
        .sort([text_column]) \
        .join(operations.InnerJoiner(suffix_enc, suffix_all), graph2, [text_column]) \
        .map(operations.InverseFrequency(frequency_column + suffix_enc, frequency_column + suffix_all, result_column)) \
        .sort([doc_column, result_column]) \
        .reduce(operations.TopN(result_column, 10, ascending=True), [doc_column]) \
        .map(operations.Project([doc_column, text_column, result_column]))

    return graph3


def pmi_graph_from_file(input_stream_name: str, parser: tp.Callable[[str], operations.TRow], doc_column: str = 'doc_id',
                        text_column: str = 'text',
                        result_column: str = 'pmi') -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    count_column = "words_amount"

    def words_filter(row: operations.TRow) -> bool:
        return len(row[text_column]) > 4 and row[count_column] > 1

    graph1 = Graph.graph_from_file(input_stream_name, parser) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([doc_column, text_column])

    filter_graph = graph1 \
        .reduce(operations.Count(count_column), [doc_column, text_column]) \
        .map(operations.Filter(words_filter))

    filtered_graph = graph1.join(operations.InnerJoiner(), filter_graph, [doc_column, text_column])

    frequency_column = "words_frequency"
    graph2 = filtered_graph.reduce(operations.TermFrequency(text_column, frequency_column), []) \
        .sort([text_column])

    suffix_enc = ""
    suffix_all = "_overall"

    graph3 = filtered_graph.sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column, frequency_column), [doc_column]) \
        .sort([text_column]) \
        .join(operations.InnerJoiner(suffix_enc, suffix_all), graph2, [text_column]) \
        .map(operations.InverseFrequency(frequency_column + suffix_enc, frequency_column + suffix_all, result_column)) \
        .sort([doc_column, result_column]) \
        .reduce(operations.TopN(result_column, 10, ascending=True), [doc_column]) \
        .map(operations.Project([doc_column, text_column, result_column]))

    return graph3


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed') -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    distance_column = "length"
    coord_graph = Graph.graph_from_iter(input_stream_name_length) \
        .map(operations.CalculateDistance(start_coord_column, end_coord_column, distance_column)) \
        .sort([edge_id_column])

    time_delta_column = "time_delta"
    time_graph = Graph.graph_from_iter(input_stream_name_time) \
        .map(operations.WeekDay(enter_time_column, weekday_result_column)) \
        .map(operations.Hour(enter_time_column, hour_result_column)) \
        .map(operations.TimeDelta(enter_time_column, leave_time_column, time_delta_column)) \
        .sort([edge_id_column]) \
        .join(operations.InnerJoiner(), coord_graph, [edge_id_column]) \
        .map(operations.Speed(distance_column, time_delta_column, speed_result_column)) \
        .sort([weekday_result_column, hour_result_column]) \
        .reduce(operations.Mean(speed_result_column), [weekday_result_column, hour_result_column]) \
        .map(operations.Project([weekday_result_column, hour_result_column, speed_result_column]))

    return time_graph


def yandex_maps_graph_from_file(input_stream_name_time: str, input_stream_name_length: str,
                                parser: tp.Callable[[str], operations.TRow],
                                enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                                edge_id_column: str = 'edge_id', start_coord_column: str = 'start',
                                end_coord_column: str = 'end',
                                weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                                speed_result_column: str = 'speed') -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    distance_column = "length"
    coord_graph = Graph.graph_from_file(input_stream_name_length, parser) \
        .map(operations.CalculateDistance(start_coord_column, end_coord_column, distance_column)) \
        .map(operations.Project([edge_id_column, distance_column])) \
        .sort([edge_id_column])

    time_delta_column = "time_delta"
    time_graph = Graph.graph_from_file(input_stream_name_time, parser) \
        .map(operations.WeekDay(enter_time_column, weekday_result_column)) \
        .map(operations.Hour(enter_time_column, hour_result_column)) \
        .map(operations.TimeDelta(enter_time_column, leave_time_column, time_delta_column)) \
        .map(operations.Project([edge_id_column, time_delta_column, weekday_result_column, hour_result_column])) \
        .sort([edge_id_column]) \
        .join(operations.InnerJoiner(), coord_graph, [edge_id_column]) \
        .map(operations.Speed(distance_column, time_delta_column, speed_result_column)) \
        .sort([weekday_result_column, hour_result_column]) \
        .reduce(operations.Mean(speed_result_column), [weekday_result_column, hour_result_column]) \
        .map(operations.Project([weekday_result_column, hour_result_column, speed_result_column]))

    return time_graph
