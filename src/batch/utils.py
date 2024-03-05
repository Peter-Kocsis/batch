def split_list(tensor, chunk_size, dim=0):
    assert dim == 0, "Only dim=0 split is supported for now"
    return [tensor[i:i + chunk_size] for i in range(0, len(tensor), chunk_size)]
