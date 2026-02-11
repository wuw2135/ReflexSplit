from models.arch.reflexsplit import ReflexSplit


def reflex_large(args):
    enc_blk_nums = [12, 8, 4, 2, 2]
    dec_blk_nums = [2, 2, 2, 2, 2]

    return ReflexSplit(args, input_resolution=(384, 384), window_size=12,
                enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
