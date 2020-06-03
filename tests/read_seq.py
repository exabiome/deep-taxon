if __name__ == '__main__':
    from exabiome.nn.train import parse_args, run, check_model, get_dataset
    import sys

    input_path = sys.argv[1]


    # test that slicing with window works on label encoded sequences
    dataset, io = get_dataset(input_path)

    idx1, seq1, label1 = dataset.difile[0]
    idx2, seq2, label2 = dataset.difile[1]

    print(len(seq1), len(seq2))

    io.close()

    dataset, io = get_dataset(input_path, window=100, step=100)

    import numpy as np
    for i in [0, 200, 248]:
        print(i)
        idx_i, seq_i, label_i = dataset.difile[i]
        assert label_i == label1
        np.testing.assert_array_equal(seq_i, seq1[i*100:(i+1)*100])

    for i in [249, 320, 321]:
        print(i)
        idx_i, seq_i, label_i = dataset.difile[i]
        assert label_i == label2
        i = i - 249
        np.testing.assert_array_equal(seq_i, seq2[i*100:(i+1)*100])

    io.close()

    # test that slicing with window works on ONE-HOT encoded sequences
    dataset, io = get_dataset(input_path)

    idx1, seq1, label1 = dataset[0]
    idx2, seq2, label2 = dataset[1]

    io.close()

    dataset, io = get_dataset(input_path, window=100, step=100)

    import numpy as np
    for i in [0, 200, 248]:
        print(i)
        idx_i, seq_i, label_i = dataset[i]
        assert label_i == label1
        np.testing.assert_array_equal(seq_i, seq1[:,i*100:(i+1)*100])

    for i in [249, 320, 321]:
        print(i)
        idx_i, seq_i, label_i = dataset[i]
        assert label_i == label2
        i = i - 249
        np.testing.assert_array_equal(seq_i, seq2[:,i*100:(i+1)*100])
