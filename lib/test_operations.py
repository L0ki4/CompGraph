from operator import itemgetter

from pytest import approx

from . import operations as ops


def test_dummy_map() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 2, 'text': 'testing out stuff'}
    ]

    result = ops.Map(ops.DummyMapper())

    assert tests == list(result(tests))


def test_lower_case() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'camelCaseTest'},
        {'test_id': 2, 'text': 'UPPER_CASE_TEST'},
        {'test_id': 3, 'text': 'wEiRdTeSt'}
    ]

    expected: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'camelcasetest'},
        {'test_id': 2, 'text': 'upper_case_test'},
        {'test_id': 3, 'text': 'weirdtest'}
    ]

    result = ops.Map(ops.LowerCase(column='text'))(tests)

    assert expected == list(result)


def test_filtering_punctuation() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'Hello, world!'},
        {'test_id': 2, 'text': 'Test. with. a. lot. of. dots.'},
        {'test_id': 3, 'text': r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'}
    ]

    expected: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'Hello world'},
        {'test_id': 2, 'text': 'Test with a lot of dots'},
        {'test_id': 3, 'text': ''}
    ]

    result = ops.Map(ops.FilterPunctuation(column='text'))(tests)

    assert expected == list(result)


def test_splitting() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'one two three'},
        {'test_id': 2, 'text': 'tab\tsplitting\ttest'},
        {'test_id': 3, 'text': 'more\nlines\ntest'},
        {'test_id': 4, 'text': 'tricky\u00A0test'}
    ]

    expected: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'one'},
        {'test_id': 1, 'text': 'three'},
        {'test_id': 1, 'text': 'two'},

        {'test_id': 2, 'text': 'splitting'},
        {'test_id': 2, 'text': 'tab'},
        {'test_id': 2, 'text': 'test'},

        {'test_id': 3, 'text': 'lines'},
        {'test_id': 3, 'text': 'more'},
        {'test_id': 3, 'text': 'test'},

        {'test_id': 4, 'text': 'test'},
        {'test_id': 4, 'text': 'tricky'}
    ]

    result = ops.Map(ops.Split(column='text'))(tests)

    assert expected == sorted(result, key=itemgetter('test_id', 'text'))


def test_product() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'speed': 5, 'distance': 10},
        {'test_id': 2, 'speed': 60, 'distance': 2},
        {'test_id': 3, 'speed': 3, 'distance': 15},
        {'test_id': 4, 'speed': 100, 'distance': 0.5},
        {'test_id': 5, 'speed': 48, 'distance': 15},
    ]

    expected: ops.TRowsIterable = [
        {'test_id': 1, 'speed': 5, 'distance': 10, 'time': 50},
        {'test_id': 2, 'speed': 60, 'distance': 2, 'time': 120},
        {'test_id': 3, 'speed': 3, 'distance': 15, 'time': 45},
        {'test_id': 4, 'speed': 100, 'distance': 0.5, 'time': 50},
        {'test_id': 5, 'speed': 48, 'distance': 15, 'time': 720},
    ]

    result = ops.Map(ops.Product(columns=['speed', 'distance'], result_column='time'))(tests)

    assert expected == list(result)


def test_filter() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'f': 0, 'g': 0},
        {'test_id': 2, 'f': 0, 'g': 1},
        {'test_id': 3, 'f': 1, 'g': 0},
        {'test_id': 4, 'f': 1, 'g': 1}
    ]

    expected: ops.TRowsIterable = [
        {'test_id': 2, 'f': 0, 'g': 1},
        {'test_id': 3, 'f': 1, 'g': 0}
    ]

    def xor(row: ops.TRow) -> bool:
        return row['f'] ^ row['g']

    result = ops.Map(ops.Filter(condition=xor))(tests)

    assert expected == list(result)


def test_projection() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'junk': 'x', 'value': 42},
        {'test_id': 2, 'junk': 'y', 'value': 1},
        {'test_id': 3, 'junk': 'z', 'value': 144}
    ]

    expected: ops.TRowsIterable = [
        {'value': 42},
        {'value': 1},
        {'value': 144}
    ]

    result = ops.Map(ops.Project(columns=['value']))(tests)

    assert expected == list(result)


def test_dummy_reduce() -> None:
    tests: ops.TRowsIterable = [
        {'test_id': 1, 'text': 'hello, world'},
        {'test_id': 2, 'text': 'bye!'}
    ]

    result = ops.Reduce(ops.FirstReducer(), keys=['test_id'])(tests)

    assert tests == list(result)


def test_top_n() -> None:
    matches: ops.TRowsIterable = [
        {'match_id': 1, 'player_id': 1, 'rank': 42},
        {'match_id': 1, 'player_id': 2, 'rank': 7},
        {'match_id': 1, 'player_id': 3, 'rank': 0},
        {'match_id': 1, 'player_id': 4, 'rank': 39},

        {'match_id': 2, 'player_id': 5, 'rank': 15},
        {'match_id': 2, 'player_id': 6, 'rank': 39},
        {'match_id': 2, 'player_id': 7, 'rank': 27},
        {'match_id': 2, 'player_id': 8, 'rank': 7}
    ]

    expected: ops.TRowsIterable = [
        {'match_id': 1, 'player_id': 1, 'rank': 42},
        {'match_id': 1, 'player_id': 2, 'rank': 7},
        {'match_id': 1, 'player_id': 4, 'rank': 39},

        {'match_id': 2, 'player_id': 5, 'rank': 15},
        {'match_id': 2, 'player_id': 6, 'rank': 39},
        {'match_id': 2, 'player_id': 7, 'rank': 27}
    ]

    presorted_matches = sorted(matches, key=itemgetter('match_id'))  # !!!
    result = ops.Reduce(ops.TopN(column='rank', n=3), keys=['match_id'])(presorted_matches)

    assert expected == sorted(result, key=itemgetter('match_id', 'player_id'))


def test_term_frequency() -> None:
    docs: ops.TRowsIterable = [
        {'doc_id': 1, 'text': 'hello', 'count': 1},
        {'doc_id': 1, 'text': 'little', 'count': 1},
        {'doc_id': 1, 'text': 'world', 'count': 1},

        {'doc_id': 2, 'text': 'little', 'count': 1},

        {'doc_id': 3, 'text': 'little', 'count': 3},
        {'doc_id': 3, 'text': 'little', 'count': 3},
        {'doc_id': 3, 'text': 'little', 'count': 3},

        {'doc_id': 4, 'text': 'little', 'count': 2},
        {'doc_id': 4, 'text': 'hello', 'count': 1},
        {'doc_id': 4, 'text': 'little', 'count': 2},
        {'doc_id': 4, 'text': 'world', 'count': 1},

        {'doc_id': 5, 'text': 'hello', 'count': 2},
        {'doc_id': 5, 'text': 'hello', 'count': 2},
        {'doc_id': 5, 'text': 'world', 'count': 1},

        {'doc_id': 6, 'text': 'world', 'count': 4},
        {'doc_id': 6, 'text': 'world', 'count': 4},
        {'doc_id': 6, 'text': 'world', 'count': 4},
        {'doc_id': 6, 'text': 'world', 'count': 4},
        {'doc_id': 6, 'text': 'hello', 'count': 1}
    ]

    expected: ops.TRowsIterable = [
        {'doc_id': 1, 'text': 'hello', 'tf': approx(0.3333, abs=0.001)},
        {'doc_id': 1, 'text': 'little', 'tf': approx(0.3333, abs=0.001)},
        {'doc_id': 1, 'text': 'world', 'tf': approx(0.3333, abs=0.001)},

        {'doc_id': 2, 'text': 'little', 'tf': approx(1.0)},

        {'doc_id': 3, 'text': 'little', 'tf': approx(1.0)},

        {'doc_id': 4, 'text': 'hello', 'tf': approx(0.25)},
        {'doc_id': 4, 'text': 'little', 'tf': approx(0.5)},
        {'doc_id': 4, 'text': 'world', 'tf': approx(0.25)},

        {'doc_id': 5, 'text': 'hello', 'tf': approx(0.666, abs=0.001)},
        {'doc_id': 5, 'text': 'world', 'tf': approx(0.333, abs=0.001)},

        {'doc_id': 6, 'text': 'hello', 'tf': approx(0.2)},
        {'doc_id': 6, 'text': 'world', 'tf': approx(0.8)}
    ]

    presorted_docs = sorted(docs, key=itemgetter('doc_id'))  # !!!
    result = ops.Reduce(ops.TermFrequency(words_column='text'), keys=['doc_id'])(presorted_docs)

    assert expected == sorted(result, key=itemgetter('doc_id', 'text'))


def test_counting() -> None:
    sentences: ops.TRowsIterable = [
        {'sentence_id': 1, 'word': 'hello'},
        {'sentence_id': 1, 'word': 'my'},
        {'sentence_id': 1, 'word': 'little'},
        {'sentence_id': 1, 'word': 'world'},

        {'sentence_id': 2, 'word': 'hello'},
        {'sentence_id': 2, 'word': 'my'},
        {'sentence_id': 2, 'word': 'little'},
        {'sentence_id': 2, 'word': 'little'},
        {'sentence_id': 2, 'word': 'hell'}
    ]

    expected: ops.TRowsIterable = [
        {'count': 1, 'word': 'hell'},
        {'count': 1, 'word': 'world'},
        {'count': 2, 'word': 'hello'},
        {'count': 2, 'word': 'my'},
        {'count': 3, 'word': 'little'}
    ]

    presorted_words = sorted(sentences, key=itemgetter('word'))  # !!!
    result = ops.Reduce(ops.Count(column='count'), keys=['word'])(presorted_words)

    assert expected == sorted(result, key=itemgetter('count', 'word'))


def test_sum() -> None:
    matches: ops.TRowsIterable = [
        {'match_id': 1, 'player_id': 1, 'score': 42},
        {'match_id': 1, 'player_id': 2, 'score': 7},
        {'match_id': 1, 'player_id': 3, 'score': 0},
        {'match_id': 1, 'player_id': 4, 'score': 39},

        {'match_id': 2, 'player_id': 5, 'score': 15},
        {'match_id': 2, 'player_id': 6, 'score': 39},
        {'match_id': 2, 'player_id': 7, 'score': 27},
        {'match_id': 2, 'player_id': 8, 'score': 7}
    ]

    expected: ops.TRowsIterable = [
        {'match_id': 1, 'score': 88},
        {'match_id': 2, 'score': 88}
    ]

    presorted_matches = sorted(matches, key=itemgetter('match_id'))  # !!!
    result = ops.Reduce(ops.Sum(column='score'), keys=['match_id'])(presorted_matches)

    assert expected == sorted(result, key=itemgetter('match_id'))


def test_simple_join() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 1, 'username': 'XeroX'},
        {'player_id': 2, 'username': 'jay'},
        {'player_id': 3, 'username': 'Destroyer'},
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 99},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 1, 'score': 22}
    ]

    expected: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 99, 'username': 'Destroyer'},
        {'game_id': 2, 'player_id': 1, 'score': 17, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 1, 'score': 22, 'username': 'XeroX'}
    ]

    presorted_games = sorted(games, key=itemgetter('player_id'))  # !!!
    presorted_players = sorted(players, key=itemgetter('player_id'))  # !!!
    result = ops.Join(ops.InnerJoiner(), keys=['player_id'])(presorted_games, presorted_players)

    assert expected == sorted(result, key=itemgetter('game_id'))


def test_inner_join() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 0, 'username': 'root'},
        {'player_id': 1, 'username': 'XeroX'},
        {'player_id': 2, 'username': 'jay'}
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 9999999},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 2, 'score': 22}
    ]

    expected: ops.TRowsIterable = [
        # player 3 is unknown
        # no games for player 0
        {'game_id': 2, 'player_id': 1, 'score': 17, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 2, 'score': 22, 'username': 'jay'}
    ]

    presorted_games = sorted(games, key=itemgetter('player_id'))  # !!!
    presorted_players = sorted(players, key=itemgetter('player_id'))  # !!!
    result = ops.Join(ops.InnerJoiner(), keys=['player_id'])(presorted_games, presorted_players)

    assert expected == sorted(result, key=itemgetter('game_id'))


def test_outer_join() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 0, 'username': 'root'},
        {'player_id': 1, 'username': 'XeroX'},
        {'player_id': 2, 'username': 'jay'}
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 9999999},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 2, 'score': 22}
    ]

    expected: ops.TRowsIterable = [
        {'player_id': 0, 'username': 'root'},  # no such game
        {'game_id': 1, 'player_id': 3, 'score': 9999999},  # no such player
        {'game_id': 2, 'player_id': 1, 'score': 17, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 2, 'score': 22, 'username': 'jay'}
    ]

    presorted_games = sorted(games, key=itemgetter('player_id'))  # !!!
    presorted_players = sorted(players, key=itemgetter('player_id'))  # !!!
    result = ops.Join(ops.OuterJoiner(), keys=['player_id'])(presorted_games, presorted_players)

    assert expected == sorted(result, key=lambda x: x.get('game_id', -1))


def test_left_join() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 0, 'username': 'root'},
        {'player_id': 1, 'username': 'XeroX'},
        {'player_id': 2, 'username': 'jay'}
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 0},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 2, 'score': 22},
        {'game_id': 4, 'player_id': 2, 'score': 41}
    ]

    expected: ops.TRowsIterable = [
        # ignore player 0 with 0 games
        {'game_id': 1, 'player_id': 3, 'score': 0},  # unknown player 3
        {'game_id': 2, 'player_id': 1, 'score': 17, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 2, 'score': 22, 'username': 'jay'},
        {'game_id': 4, 'player_id': 2, 'score': 41, 'username': 'jay'}
    ]

    presorted_games = sorted(games, key=itemgetter('player_id'))  # !!!
    presorted_players = sorted(players, key=itemgetter('player_id'))  # !!!
    result = ops.Join(ops.LeftJoiner(), keys=['player_id'])(presorted_games, presorted_players)

    assert expected == sorted(result, key=itemgetter('game_id'))


def test_right_join() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 0, 'username': 'root'},
        {'player_id': 1, 'username': 'XeroX'},
        {'player_id': 2, 'username': 'jay'}
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 0},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 2, 'score': 22},
        {'game_id': 4, 'player_id': 2, 'score': 41},
        {'game_id': 5, 'player_id': 1, 'score': 34}
    ]

    expected: ops.TRowsIterable = [
        # ignore game with unknown player 3
        {'player_id': 0, 'username': 'root'},  # no games for root
        {'game_id': 2, 'player_id': 1, 'score': 17, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 2, 'score': 22, 'username': 'jay'},
        {'game_id': 4, 'player_id': 2, 'score': 41, 'username': 'jay'},
        {'game_id': 5, 'player_id': 1, 'score': 34, 'username': 'XeroX'}
    ]

    presorted_games = sorted(games, key=itemgetter('player_id'))  # !!!
    presorted_players = sorted(players, key=itemgetter('player_id'))  # !!!
    result = ops.Join(ops.RightJoiner(), keys=['player_id'])(presorted_games, presorted_players)

    assert expected == sorted(result, key=lambda x: x.get('game_id', -1))


def test_simple_join_with_collision() -> None:
    players: ops.TRowsIterable = [
        {'player_id': 1, 'username': 'XeroX', 'score': 400},
        {'player_id': 2, 'username': 'jay', 'score': 451},
        {'player_id': 3, 'username': 'Destroyer', 'score': 999},
    ]

    games: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score': 99},
        {'game_id': 2, 'player_id': 1, 'score': 17},
        {'game_id': 3, 'player_id': 1, 'score': 22}
    ]

    expected: ops.TRowsIterable = [
        {'game_id': 1, 'player_id': 3, 'score_game': 99, 'score_max': 999, 'username': 'Destroyer'},
        {'game_id': 2, 'player_id': 1, 'score_game': 17, 'score_max': 400, 'username': 'XeroX'},
        {'game_id': 3, 'player_id': 1, 'score_game': 22, 'score_max': 400, 'username': 'XeroX'}
    ]

    presorted_games = sorted(games, key=itemgetter('player_id'))  # !!!
    presorted_players = sorted(players, key=itemgetter('player_id'))  # !!!
    result = ops.Join(ops.InnerJoiner(suffix_a='_game', suffix_b='_max'),
                      keys=['player_id'])(presorted_games, presorted_players)

    assert expected == sorted(result, key=itemgetter('game_id'))
