from collections.abc import Iterator


def _positions(limit: int, tile: int, stride: int):
    # all regular positions
    last_start = limit - tile
    if last_start < 0:
        return [0]  # tile bigger than image => single tile anchored at 0
    xs = list(range(0, last_start + 1, stride))
    if xs[-1] != last_start:
        xs.append(last_start)
    return xs


def gen_tiles_cover(h: int, w: int, tile_h: int, tile_w: int, stride_h: int, stride_w: int) -> Iterator[tuple[int, int, int, int, int, int]]:
    ys = _positions(h, tile_h, stride_h)
    xs = _positions(w, tile_w, stride_w)

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            yield x, y, x + tile_w, y + tile_h, row, col
