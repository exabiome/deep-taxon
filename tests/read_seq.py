if __name__ == '__main__':
    from exabiome.nn.train import parse_args, run, check_model, get_dataset

    model, input_path, criterion, args = parse_args()

    dataset, io = get_dataset(input_path)

    idx1, seq1, label1 = dataset.difile[0]
    idx2, seq2, label2 = dataset.difile[1]

    io.close()

    dataset, io = get_dataset(input_path, window=100, step=100)

    import numpy as np
    for i in [0, 1, 3]:
        print(i)
        idx_i, seq_i, label_i = dataset.difile[i]
        assert label_i == label1
        np.testing.assert_array_equal(seq_i, seq1[i*100:(i+1)*100])

    for i in [4, 5, 7]:
        print(i)
        idx_i, seq_i, label_i = dataset.difile[i]
        assert label_i == label2
        #breakpoint()
        i = i - 4
        np.testing.assert_array_equal(seq_i, seq2[i*100:(i+1)*100])
