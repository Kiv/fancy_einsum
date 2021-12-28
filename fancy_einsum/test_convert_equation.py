from fancy_einsum import convert_equation


def test_simple_matmul():
    actual = convert_equation('rows temp, temp cols -> rows cols')
    assert actual == 'rt,tc->rc'


def test_infer_rhs_trace():
    actual = convert_equation('arr arr ')
    assert actual == 'aa->'


def test_sum():
    actual = convert_equation('arr -> ')
    assert actual == 'a->'


def test_elemwise():
    actual = convert_equation(
        'batch rows cols, batch rows cols -> batch rows cols')
    assert actual == 'brc,brc->brc'


def test_elemwise_transpose():
    actual = convert_equation('rows cols, cols rows -> rows cols')
    assert actual == 'rc,cr->rc'


def test_ellipse():
    actual = convert_equation('...vec, ...vec -> ...')
    assert actual == '...v,...v->...'


def test_ellipse_matmul():
    actual = convert_equation('... rows temp, ... temp cols -> ... rows cols')
    assert actual == '...rt,...tc->...rc'


def test_conflicts():
    actual = convert_equation(
        'batch channels time1, batch channels time2 -> batch channels time1 time2')
    assert actual == 'bct,bcT->bctT'


def test_conflicts_infer():
    actual = convert_equation('batch channels time1, batch channels time2')
    assert actual == 'bct,bcT->tT'


def test_sum_all():
    actual = convert_equation('...,...')
    assert actual == '...,...->...'


def test_batch_mm():
    actual = convert_equation(
        'batch rows t, batch t columns -> batch rows columns')
    assert actual == 'brt,btc->brc'


def test_transpose_infer():
    actual = convert_equation('batch y x')
    assert actual == 'byx->bxy'


def test_extra_spaces():
    actual = convert_equation('  a b,  b c ->  a c  ')
    assert actual == 'ab,bc->ac'


def test_chain_matmul():
    actual = convert_equation(
        'rows t1, t1 t2, t2 t3, t3 t4, t4 cols -> rows cols')
    assert actual == 'rt,tT,TA,AB,Bc->rc'
