from . import graphs
import pathlib

path = pathlib.Path(__file__).parent


def test_word_count_file_run() -> None:
    graph = graphs.word_count_graph_from_file(str(path) + '/resource/text_corpus.txt', parser=lambda x: eval(x),
                                              text_column='text',
                                              count_column='count')

    _ = graph.run()


def test_tf_idf_file_run() -> None:
    graph = graphs.inverted_index_graph_from_file(str(path) + '/resource/text_corpus.txt', parser=lambda x: eval(x),
                                                  doc_column='doc_id', text_column='text', result_column='tf_idf')

    _ = graph.run()


def test_pmi_file_run() -> None:
    graph = graphs.pmi_graph_from_file(str(path) + '/resource/text_corpus.txt', parser=lambda x: eval(x),
                                       doc_column='doc_id',
                                       text_column='text', result_column='pmi')

    _ = graph.run()


def test_yandex_maps_heavy_file_run() -> None:
    graph = graphs.yandex_maps_graph_from_file(
        str(path) + '/resource/travel_times.txt', str(path) + '/resource/road_graph_data.txt', parser=lambda x: eval(x),
        enter_time_column='enter_time', leave_time_column='leave_time', edge_id_column='edge_id',
        start_coord_column='start', end_coord_column='end',
        weekday_result_column='weekday', hour_result_column='hour', speed_result_column='speed'
    )

    _ = graph.run()
    # df = pd.DataFrame(res)

    # plt.figure(figsize=(16, 10))
    # sns.lineplot(data=df, x='hour', y='speed', hue='weekday')
    # plt.legend(loc=4)
    # plt.xticks(np.arange(0, 24, 1))
    # plt.grid()

    # plt.savefig("compgraph/resource/plot_task_4.png", dpi=100)
