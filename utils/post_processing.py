import operator

def update_split_block_phase_sets(solutions, idx2var):
    """
    For variants we do not phase (haplotype "h" field set to 2) we need to update the phase sets of variants downstream in the block
    since the block is now split, and also need to handle the case of multiple unphased variants within a block.
    """
    max_phase_set = len(solutions)
    sorted_vars = sorted(idx2var.items(), key=operator.itemgetter(0))
    change_set = False
    cur_phase_set = 0
    for i in sorted_vars:
        if change_set and i[1].phase_set != cur_phase_set: 
            # end of current block
            change_set = False
            max_phase_set += 1
        if change_set:
            if i[1].h == 2: 
                # hit another unphased variant in the same block, so need to change phase set again
                max_phase_set += 1
            else:
                i[1].phase_set = max_phase_set
        if i[1].h == 2: 
            # hit a variant that shouldn't be phased, so change the phase set of subsequent variants
            cur_phase_set = i[1].phase_set
            change_set = True
    return dict(sorted_vars)
